"""Press-message view and MESSAGES block for Diplomacy press games.

A transition row of data/processed/diplomacy_press/transitions.parquet carries `msgs_json`: the messages of that phase, already restricted at
extraction time (press_build.py transitions) to those stored in the phase's own container, labelled with the phase's own name, and sent strictly
before the first message of any later phase. Each message is [sender, recipient, time_sent, raw_text]. Games whose dump stores every message in the
final container (no message in any transition container) are not in transitions.parquet (press_build.py index -> games_excluded.parquet).

view(msgs, power)        -> the messages the acting power could see: sender == power, recipient == power, or recipient == GLOBAL; time_sent order
clean(text)              -> HTML stripped (tags removed, <br> -> space, entities unescaped), whitespace collapsed
render_block(...)        -> "MESSAGES" block, one line per message "<SND> -> <RCP>: <text>" (power abbreviations of eventworld.dip_serialize; GLOBAL -> ALL),
                            capped at a token budget (Qwen3 tokenizer) keeping the most recent messages
insert_block(prompt, b)  -> the Task S prompt with the block inserted right before its final OWN line (so S and SM differ only by the block)
"""
import html, re, sys
from eventworld.dip_serialize import ABBR

TAG = re.compile(r"</?[A-Za-z][^<>]{0,300}>")
BR = re.compile(r"<\s*br\s*/?\s*>", re.I)
WS = re.compile(r"\s+")
BUDGET = 1500        # tokens for the whole MESSAGES block (header + lines)
MSG_CAP = 400        # tokens for a single message line; longer texts are cut and end with " ..."


def clean(text):
    t = BR.sub(" ", text or "")
    t = TAG.sub(" ", t)
    t = html.unescape(t)
    return WS.sub(" ", t).strip()


SPECIAL = {"GLOBAL": "ALL", "MASTER": "GM", "SYSTEM": "SYSTEM"}   # broadcast; game master / system notices (~2k messages in the dump)


def abbr(p):
    return SPECIAL.get(p) or ABBR.get(p, str(p)[:3].upper())


def view(msgs, power):
    """msgs: list of [sender, recipient, time_sent, text] (stored order). Stable sort by time_sent."""
    sel = [(i, m) for i, m in enumerate(msgs) if m[0] == power or m[1] == power or m[1] == "GLOBAL"]
    return [m for _, m in sorted(sel, key=lambda x: (x[1][2], x[0]))]


class Renderer:
    """Holds the tokenizer; render() returns (block_text, info) with info = n_msgs, n_kept, n_cut_msgs, tokens."""

    def __init__(self, tokenizer_name="Qwen/Qwen3-1.7B", budget=BUDGET, msg_cap=MSG_CAP):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.budget, self.msg_cap = budget, msg_cap

    def ntok(self, texts):
        if not texts:
            return []
        return [len(x) for x in self.tok(list(texts), add_special_tokens=False)["input_ids"]]

    def _cap_line(self, line, n):
        if n <= self.msg_cap:
            return line, False
        ids = self.tok(line, add_special_tokens=False)["input_ids"][: self.msg_cap - 2]
        return self.tok.decode(ids).rstrip() + " ...", True

    def lines(self, msgs, power, prior=None):
        """prior: optional list of (phase_name, msgs) for earlier phases (message lookback, oldest first); their lines carry a [phase] prefix."""
        out = []
        for ph, ms in (prior or []):
            out += [f"[{ph}] {abbr(m[0])} -> {abbr(m[1])}: {clean(m[3])}" for m in view(ms, power)]
        out += [f"{abbr(m[0])} -> {abbr(m[1])}: {clean(m[3])}" for m in view(msgs, power)]
        return [l for l in out if not l.endswith(": ")]   # messages that are empty after cleaning are dropped

    def render(self, msgs, power, prior=None):
        ls = self.lines(msgs, power, prior)
        n_all = len(ls)
        if not ls:
            return "MESSAGES\nnone", dict(n_msgs=0, n_kept=0, n_cut_msgs=0, truncated=False, tokens=self.ntok(["MESSAGES\nnone\n"])[0])
        counts = self.ntok([l + "\n" for l in ls])
        cut = 0
        for i, n in enumerate(counts):
            if n > self.msg_cap:
                ls[i], _ = self._cap_line(ls[i], n); counts[i] = self.ntok([ls[i] + "\n"])[0]; cut += 1
        head_tok = 12  # "MESSAGES (NNN earlier omitted)\n" upper bound
        kept, used = [], head_tok
        for l, n in zip(reversed(ls), reversed(counts)):
            if used + n > self.budget:
                break
            kept.append(l); used += n
        kept.reverse()
        omitted = n_all - len(kept)
        head = "MESSAGES" if not omitted else f"MESSAGES ({omitted} earlier omitted)"
        block = "\n".join([head] + kept)
        return block, dict(n_msgs=n_all, n_kept=len(kept), n_cut_msgs=cut, truncated=bool(omitted), tokens=self.ntok([block + "\n"])[0])


def insert_block(prompt, block):
    lines = prompt.split("\n")
    assert lines[-1].startswith("OWN "), lines[-1]
    return "\n".join(lines[:-1] + [block, lines[-1]])


def strip_block(prompt):
    """Inverse of insert_block (for the S-vs-SM identity check): drop the lines from 'MESSAGES' up to (not including) the OWN line."""
    lines = prompt.split("\n")
    i = next(k for k, l in enumerate(lines) if l == "MESSAGES" or l.startswith("MESSAGES ("))
    return "\n".join(lines[:i] + [lines[-1]])
