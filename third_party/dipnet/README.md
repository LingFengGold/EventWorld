# DipNet

The Diplomacy baselines, the opponents of the agent experiments, and the agents themselves use the pretrained DipNet policies of
Paquette et al. (2019, github.com/diplomacy/research): the supervised policy (SL) and its RL fine-tuned version (RL). `eventworld/dipnet.py`
calls them from any Python; it needs the following under this directory (or under `$DIPNET_HOME`):

| Path | Content |
|---|---|
| `research/` | a clone of github.com/diplomacy/research (commit `0ce19e2`) with its gRPC stubs (`*_pb2_grpc.py`) generated |
| `env37/` | Python 3.7 with `tensorflow==1.13.1`, `grpcio==1.15`, `protobuf==3.6.1`, `diplomacy==1.1.0` and the requirements of `research/` |
| `models/sl`, `models/rl` | the unpacked `neurips2019-sl_model.zip` and `neurips2019-rl_model.zip` SavedModels (download links in the research repository) |
| `containers/tfserving.sif` | the TensorFlow Serving 1.13 image of the research repository (`tensorflow_serving-f16e777-tf1.13-001`), run with Apptainer or Singularity |

On the first call the module starts TensorFlow Serving on CPU (port 9501 for SL, 9502 for RL) and an `env37` worker that builds DipNet's
features and decodes its orders; both are reused afterwards. Smoke test: `python eventworld/dipnet.py` prints opening orders for the seven powers.
Greedy decoding (`temperature=0.0`) is deterministic and is used for every opponent in the paper.
