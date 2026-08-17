from __future__ import annotations

import torch


def main() -> None:
    q = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    k = torch.arange(32, dtype=torch.float32).reshape(4, 8)

    # Q = [[q00, q01],
    #      [q10, q11]]
    q00 = q[:2, :2]
    q01 = q[:2, 2:]
    q10 = q[2:, :2]
    q11 = q[2:, 2:]

    # K = [[k00, k01, k02, k03],
    #      [k10, k11, k12, k13]]
    k00 = k[:2, :2]
    k01 = k[:2, 2:4]
    k02 = k[:2, 4:6]
    k03 = k[:2, 6:]
    k10 = k[2:, :2]
    k11 = k[2:, 2:4]
    k12 = k[2:, 4:6]
    k13 = k[2:, 6:]

    # C = Q @ K = [[c00, c01, c02, c03],
    #              [c10, c11, c12, c13]]
    c00 = q00 @ k00 + q01 @ k10
    c01 = q00 @ k01 + q01 @ k11
    c02 = q00 @ k02 + q01 @ k12
    c03 = q00 @ k03 + q01 @ k13
    c10 = q10 @ k00 + q11 @ k10
    c11 = q10 @ k01 + q11 @ k11
    c12 = q10 @ k02 + q11 @ k12
    c13 = q10 @ k03 + q11 @ k13

    top_row = torch.cat((c00, c01, c02, c03), dim=1)
    bottom_row = torch.cat((c10, c11, c12, c13), dim=1)
    block_output = torch.cat((top_row, bottom_row), dim=0)

    torch.testing.assert_close(block_output, q @ k)
    print("PASS: block matrix multiply matches Q @ K")


if __name__ == "__main__":
    main()
