"""Smoke-test the compiled PointNet++ CUDA extension used by PointNeXt."""

import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in PyTorch")

    # Import torch first so its shared libraries (libc10, libtorch) are loaded.
    import pointnet2_batch_cuda as pointnet2_cuda

    device = torch.device("cuda")
    batch_size, num_points, num_samples = 2, 128, 32
    xyz = torch.rand(batch_size, num_points, 3, device=device).contiguous()

    temp = torch.full(
        (batch_size, num_points), 1e10, dtype=torch.float32, device=device
    )
    fps_idx = torch.empty(
        (batch_size, num_samples), dtype=torch.int32, device=device
    )
    pointnet2_cuda.furthest_point_sampling_wrapper(
        batch_size, num_points, num_samples, xyz, temp, fps_idx
    )

    query_xyz = torch.gather(
        xyz,
        1,
        fps_idx.long().unsqueeze(-1).expand(-1, -1, 3),
    ).contiguous()
    neighbors = 16
    ball_idx = torch.zeros(
        (batch_size, num_samples, neighbors), dtype=torch.int32, device=device
    )
    pointnet2_cuda.ball_query_wrapper(
        batch_size,
        num_points,
        num_samples,
        0.5,
        neighbors,
        query_xyz,
        xyz,
        ball_idx,
    )

    torch.cuda.synchronize()
    for name, indices in (("FPS", fps_idx), ("ball query", ball_idx)):
        if indices.device.type != "cuda" or indices.dtype != torch.int32:
            raise AssertionError(
                f"{name} returned device={indices.device}, dtype={indices.dtype}"
            )
        if indices.min().item() < 0 or indices.max().item() >= num_points:
            raise AssertionError(f"{name} returned an out-of-range point index")

    print(
        "pointnet2 CUDA extension: OK "
        f"(GPU={torch.cuda.get_device_name(0)}, "
        f"FPS={tuple(fps_idx.shape)}, ball_query={tuple(ball_idx.shape)})"
    )


if __name__ == "__main__":
    main()
