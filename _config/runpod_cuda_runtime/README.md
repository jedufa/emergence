# Emergence CUDA runtime image

This is the build definition for the `emergence_cuda_v1` RunPod runtime
contract. The current profile uses the immutable RunPod PyTorch/CUDA/SSH base
plus a recorded per-run setup command; this Dockerfile is not claimed to be
deployed until its derivative image digest is published and bound in the
profile. Emergence source remains a per-run allowlisted snapshot uploaded by
the runner.

Build for RunPod's x86 hosts with an explicit version tag:

```bash
docker build --platform=linux/amd64 \
  -t DOCKERHUB_USER/emergence-cuda-runtime:v1.0.0 \
  -f _config/runpod_cuda_runtime/Dockerfile \
  _config/runpod_cuda_runtime
docker push DOCKERHUB_USER/emergence-cuda-runtime:v1.0.0
```

The Docker Hub account needs a read/write personal access token. After the
push, resolve the pushed tag to its immutable digest and update the runtime
profile's `image_reference`, `image_digest`, and `definition_sha256` together.
Do not use `latest`, and do not install these packages interactively on a pod:
that would no longer be a fixed-runtime experiment.
