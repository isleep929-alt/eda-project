#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spi_xfer_public - inference_interface.py





2026/9/21/17.29
在官方 baseline 基础上修改为：
    1. 默认使用分层状态机 Agent（hfsm）
    2. 配置前主动关闭 SSIENR
    3. 配置完成后再打开 SSIENR
    4. 系统切换 Protocol / TMOD / DFS / BAUDR
    5. 按明确流程写 DR、等待真实传输、读取 DR
    6. 根据 coverage_state 的新增 bin 数决定当前组合是否再利用一次

官方接口保持不变：
    class InferenceInterface:
        def __init__(self, dut_spec_path: str, covergroup_path: str)
        def predict(self, coverage_state: np.ndarray,
                    step: int, max_steps: int) -> np.ndarray

动作空间仍为 12 维：
    [reg_we, reg_addr, reg_wdata, reg_re, rxd,
     ss_in_n, rst_n, pad0, pad1, pad2, pad3, pad4]
"""

import json
import os
from collections import deque

import numpy as np


class InferenceInterface:
    DIMS = 12

    # 保留官方动作维度上界，random baseline 仍然可以使用
    BOUNDS = np.array(
        [2, 16, 1 << 32, 2, 2, 2, 2, 1, 1, 1, 1, 1],
        dtype=np.float32
    )

    # ------------------------------------------------------------------ #
    # 【修改1】把 spi_xfer 中会用到的寄存器地址集中定义
    # 这样状态机里不需要到处写“魔法数字”
    # ------------------------------------------------------------------ #
    REG_CTRLR0 = 0
    REG_SSIENR = 2
    REG_SER = 3
    REG_BAUDR = 4
    REG_TXFTLR = 5
    REG_DR = 9

    # ------------------------------------------------------------------ #
    # 【修改2】定义 Coverage 表中关心的三种 protocol
    #
    # spi0 : FRF=0，SCPH=0
    # spi1 : FRF=0，SCPH=1
    # ssp  : FRF=1
    #
    # 公开 baseline 中：
    #   0x0807 -> SPI, TMOD0, DFS=7(8bit)
    #   0x0887 -> SPI + SCPH
    #   0x0827 -> SSP
    #
    # 所以这里保留 0x0800 作为 CTRLR0 基值。
    # ------------------------------------------------------------------ #
    CTRLR0_BASE = 0x0800

    PROTOCOL_BITS = {
        "spi0": 0x0000,
        "spi1": 0x0080,
        "ssp":  0x0020,
    }

    # ------------------------------------------------------------------ #
    # 【修改3】准备几组 float32 能精确表示的数据
    # 后面写 DR 时循环使用，避免每次都用同一个数据
    # ------------------------------------------------------------------ #
    TX_PATTERNS = (
        0x00000055,
        0x000000AA,
        0x000000FF,
        0x0000AA55,
        0x00DEADBE,
        0x00FFFFFF,
    )

    # ------------------------------------------------------------------ #
    # 保留原 greedy baseline 的候选池，方便后续和 hfsm 做对照
    # ------------------------------------------------------------------ #
    _REPEAT = 8
    _PERTURB = [(2, 0x10000)]

    _CANDIDATES = [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

        [1, 0, 0x0807, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0x0A07, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0x0C07, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0x0887, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0x0907, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0x0827, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0x0847, 0, 0, 1, 1, 0, 0, 0, 0, 0],

        [1, 3, 0x0001, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 4, 0x0001, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 4, 0x0002, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 4, 0x0003, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 4, 0x0008, 0, 0, 1, 1, 0, 0, 0, 0, 0],

        [1, 2, 0x0001, 0, 0, 1, 1, 0, 0, 0, 0, 0],

        [1, 9, 0x000000FF, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 9, 0x0000AA55, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    ]

    # ------------------------------------------------------------------ #
    # 【修改4】默认 policy 从 random 改成 hfsm
    #
    # hfsm   : 新增的分层状态机 Agent
    # random : 保留官方随机 baseline
    # greedy : 保留官方简单贪心 baseline
    # ------------------------------------------------------------------ #
    def __init__(self, dut_spec_path=None, covergroup_path=None,
                 policy="hfsm", seed=7):
        self.dut_spec_path = dut_spec_path
        self.covergroup_path = covergroup_path
        self.policy = policy
        self.rng = np.random.RandomState(seed)
        self._total_bins = self._read_total_bins()

        # 原 greedy baseline 的状态
        self._cands = deque(
            [np.asarray(c, dtype=np.float32) for c in self._CANDIDATES]
        )
        self._n = 0
        self._begin = 0

        # -------------------------------------------------------------- #
        # 【修改5】建立“高层测试计划”
        # 每一个 plan 就是一组明确的：
        # protocol + TMOD + DFS + BAUDR + TX 数据数量
        # -------------------------------------------------------------- #
        self._plans = self._build_plans()
        self._plan_idx = 0

        # 当前 plan 是否已经因为有 coverage 收益而重复过一次
        self._repeat_current = 0

        # -------------------------------------------------------------- #
        # 【修改6】建立“低层状态机”
        #
        # RESET
        # -> DISABLE
        # -> CONFIG_CTRLR0
        # -> CONFIG_SER
        # -> CONFIG_BAUDR
        # -> CONFIG_TXFTLR
        # -> ENABLE
        # -> WRITE_DR
        # -> RUN
        # -> READ_DR
        # -> FINAL_DISABLE
        # -> EVAL
        # -------------------------------------------------------------- #
        self._phase = "RESET"
        self._wait_count = 0
        self._tx_count = 0
        self._read_count = 0

        # 记录某个 plan 开始时的 coverage_state
        self._plan_cov_start = None

    # ------------------------------------------------------------------ #
    # 原官方辅助：读取 total_bins
    # ------------------------------------------------------------------ #
    def _read_total_bins(self):
        if self.covergroup_path:
            meta = os.path.join(
                os.path.dirname(os.path.abspath(self.covergroup_path)),
                "coverage_meta.json"
            )
            if os.path.exists(meta):
                try:
                    with open(meta, encoding="utf-8") as f:
                        n = int(json.load(f).get("total_bins", 0))
                    if n > 0:
                        return n
                except Exception:
                    pass
        return self.DIMS

    @property
    def total_bins(self):
        return self._total_bins

    # ------------------------------------------------------------------ #
    # 【修改7】reset() 同时重置 HFSM 和原 baseline 的内部状态
    # ------------------------------------------------------------------ #
    def reset(self):
        self._cands = deque(
            [np.asarray(c, dtype=np.float32) for c in self._CANDIDATES]
        )
        self._n = 0
        self._begin = 0

        self._plan_idx = 0
        self._repeat_current = 0
        self._phase = "RESET"
        self._wait_count = 0
        self._tx_count = 0
        self._read_count = 0
        self._plan_cov_start = None

    # ================================================================== #
    # 【修改8】Action 构造函数
    # 统一生成 12 维 np.float32 action
    # ================================================================== #
    @staticmethod
    def _action(reg_we=0, reg_addr=0, reg_wdata=0,
                reg_re=0, rxd=0, ss_in_n=1, rst_n=1):
        a = np.zeros(InferenceInterface.DIMS, dtype=np.float32)

        a[0] = np.float32(reg_we)
        a[1] = np.float32(reg_addr)
        a[2] = np.float32(reg_wdata)
        a[3] = np.float32(reg_re)
        a[4] = np.float32(rxd)
        a[5] = np.float32(ss_in_n)
        a[6] = np.float32(rst_n)

        return a

    @classmethod
    def _write_reg(cls, addr, value, rxd=0):
        """生成一次寄存器写 action。"""
        return cls._action(
            reg_we=1,
            reg_addr=addr,
            reg_wdata=value,
            reg_re=0,
            rxd=rxd,
            ss_in_n=1,
            rst_n=1,
        )

    @classmethod
    def _read_reg(cls, addr, rxd=0):
        """生成一次寄存器读 action。"""
        return cls._action(
            reg_we=0,
            reg_addr=addr,
            reg_wdata=0,
            reg_re=1,
            rxd=rxd,
            ss_in_n=1,
            rst_n=1,
        )

    @classmethod
    def _idle_action(cls, rxd=0):
        """不写、不读，让 DUT 自己继续运行一拍。"""
        return cls._action(
            reg_we=0,
            reg_addr=0,
            reg_wdata=0,
            reg_re=0,
            rxd=rxd,
            ss_in_n=1,
            rst_n=1,
        )

    @classmethod
    def _reset_action(cls):
        """rst_n 低有效，复位一拍。"""
        return cls._action(
            reg_we=0,
            reg_addr=0,
            reg_wdata=0,
            reg_re=0,
            rxd=0,
            ss_in_n=1,
            rst_n=0,
        )

    # ================================================================== #
    # 【修改9】根据 protocol / TMOD / DFS 动态生成 CTRLR0
    # 不再把所有组合全部手工写死
    # ================================================================== #
    @classmethod
    def _make_ctrlr0(cls, protocol, tmod, dfs_bits):
        """
        protocol:
            spi0 / spi1 / ssp

        tmod:
            0 / 1 / 2 / 3

        dfs_bits:
            真实有效帧宽，如 8/16/24/31/32
            寄存器字段实际写入 dfs_bits - 1

        注意：
        这里沿用公开 baseline 的 0x0800 基值。
        例如：
            spi0 + TMOD2 + DFS8
            = 0x0800 + 0x0400 + 0x0007
            = 0x0C07
        """
        if protocol not in cls.PROTOCOL_BITS:
            protocol = "spi0"

        tmod = int(np.clip(tmod, 0, 3))
        dfs_bits = int(np.clip(dfs_bits, 4, 32))
        dfs_raw = dfs_bits - 1

        return int(
            cls.CTRLR0_BASE
            | cls.PROTOCOL_BITS[protocol]
            | (tmod << 9)
            | dfs_raw
        )

    # ================================================================== #
    # 【修改10】高层 Planner：系统构造测试组合
    # ================================================================== #
    @staticmethod
    def _make_plan(name, protocol, tmod, dfs, baudr,
                   tx_words=2, ser=1, txftlr=0):
        return {
            "name": name,
            "protocol": protocol,
            "tmod": int(tmod),
            "dfs": int(dfs),
            "baudr": int(baudr),
            "tx_words": int(tx_words),
            "ser": int(ser),
            "txftlr": int(txftlr),
        }

    def _build_plans(self):
        plans = []

        # -------------------------------------------------------------- #
        # 【修改10-1】先扫 Protocol × TMOD
        # 目标：tmod_1 / tmod_2 / tmod_3
        # 同时尝试 protocol_x_tmod cross
        # -------------------------------------------------------------- #
        for protocol in ("spi0", "spi1", "ssp"):
            for tmod in (1, 2, 3):
                plans.append(
                    self._make_plan(
                        name=f"{protocol}_tmod{tmod}",
                        protocol=protocol,
                        tmod=tmod,
                        dfs=8,
                        baudr=2,
                        tx_words=2,
                    )
                )

        # -------------------------------------------------------------- #
        # 【修改10-2】再扫 Protocol × DFS
        # 优先覆盖表格里的高价值 cross
        # -------------------------------------------------------------- #
        for protocol, dfs in (
            ("spi0", 8),
            ("spi0", 32),
            ("spi1", 8),
            ("spi1", 32),
            ("ssp", 8),
            ("ssp", 16),
        ):
            plans.append(
                self._make_plan(
                    name=f"{protocol}_dfs{dfs}",
                    protocol=protocol,
                    tmod=0,
                    dfs=dfs,
                    baudr=2,
                    tx_words=2,
                )
            )

        # -------------------------------------------------------------- #
        # 【修改10-3】补 DFS 边界
        # 表格中的有效帧宽：16 / 24 / 31
        # -------------------------------------------------------------- #
        for dfs in (16, 24, 31):
            plans.append(
                self._make_plan(
                    name=f"dfs_boundary_{dfs}",
                    protocol="spi0",
                    tmod=0,
                    dfs=dfs,
                    baudr=2,
                    tx_words=2,
                )
            )

        # -------------------------------------------------------------- #
        # 【修改10-4】扫 BAUDR 边界 1 / 2 / 3 / 8
        # -------------------------------------------------------------- #
        for baudr in (1, 2, 3, 8):
            plans.append(
                self._make_plan(
                    name=f"baudr_{baudr}",
                    protocol="spi0",
                    tmod=2,
                    dfs=8,
                    baudr=baudr,
                    tx_words=2,
                )
            )

        # -------------------------------------------------------------- #
        # 【修改10-5】连续写 4 笔 DR
        # 帮助触发 TX FIFO half 以及更完整的 FSM 路径
        # -------------------------------------------------------------- #
        plans.append(
            self._make_plan(
                name="tx_fifo_half_like",
                protocol="spi0",
                tmod=0,
                dfs=8,
                baudr=2,
                tx_words=4,
            )
        )

        return plans

    @property
    def _current_plan(self):
        return self._plans[self._plan_idx]

    def _advance_plan(self):
        self._plan_idx = (self._plan_idx + 1) % len(self._plans)
        self._repeat_current = 0

    # ================================================================== #
    # 【修改11】为真实传输预留等待时间
    # 防止刚写 DR 就立即改配置
    # ================================================================== #
    @staticmethod
    def _estimate_wait_cycles(plan):
        frames = max(1, plan["tx_words"])
        dfs = max(4, plan["dfs"])
        baudr = max(1, plan["baudr"])

        wait_cycles = frames * dfs * max(2, baudr) * 3 + 48

        # 防止某一组参数独占太多时间
        return int(np.clip(wait_cycles, 96, 1600))

    # ================================================================== #
    # 【修改12】计算当前 plan 新增了多少个 bin
    # 不只是看总 coverage，而是比较前后 coverage_state
    # ================================================================== #
    def _coverage_gain(self, coverage_state):
        now = np.asarray(
            coverage_state,
            dtype=np.float32
        ).reshape(-1)

        if self._plan_cov_start is None:
            return 0

        n = min(now.size, self._plan_cov_start.size)

        now_hit = now[:n] > 0.5
        old_hit = self._plan_cov_start[:n] > 0.5

        return int(
            np.count_nonzero(
                now_hit & (~old_hit)
            )
        )

    # ------------------------------------------------------------------ #
    # 保留原 random baseline
    # ------------------------------------------------------------------ #
    def _random_action(self):
        a = (
            self.rng.uniform(0.0, 1.0, self.DIMS)
            * self.BOUNDS
        )
        return a.astype(np.float32)

    # ------------------------------------------------------------------ #
    # 保留原 greedy baseline
    # ------------------------------------------------------------------ #
    def _greedy_action(self, covered):
        if self._n == 0:
            self._begin = covered

        cand = self._cands[0]

        if self._n >= self._REPEAT:
            gain = covered - self._begin

            self._cands.rotate(-1)

            if gain > 0:
                self._cands.appendleft(cand)

            self._n = 0
            self._begin = covered

        self._n += 1

        return self._perturb(cand.copy())

    def _perturb(self, action):
        for dim, half in self._PERTURB:
            v = (
                int(action[dim])
                + self.rng.randint(-half, half + 1)
            )

            action[dim] = float(
                min(
                    max(v, 0),
                    int(self.BOUNDS[dim]) - 1
                )
            )

        return action

    # ================================================================== #
    # 【修改13】官方 predict() 接口
    #
    # 每调用一次 predict()，只输出“当前这一拍”的 action
    # 整个 SPI transaction 由多次 predict() 共同完成
    # ================================================================== #
    def predict(self, coverage_state, step, max_steps):
        coverage_state = np.asarray(
            coverage_state,
            dtype=np.float32
        ).reshape(-1)

        # -------------------------------------------------------------- #
        # 保留两个 baseline，方便比较
        # -------------------------------------------------------------- #
        if self.policy == "random":
            return self._random_action()

        if self.policy == "greedy":
            return self._greedy_action(
                int(np.sum(coverage_state))
            )

        # 默认进入 hfsm
        plan = self._current_plan

        # -------------------------------------------------------------- #
        # 状态0：RESET
        # 只在程序刚开始时复位一拍
        # -------------------------------------------------------------- #
        if self._phase == "RESET":
            self._plan_cov_start = coverage_state.copy()
            self._phase = "DISABLE"

            return self._reset_action()

        # -------------------------------------------------------------- #
        # 状态1：DISABLE
        #
        # 【关键修改】
        # 配置 CTRLR0 / SER / BAUDR / TXFTLR 前
        # 先明确写 SSIENR=0
        # -------------------------------------------------------------- #
        if self._phase == "DISABLE":
            self._phase = "CONFIG_CTRLR0"

            return self._write_reg(
                self.REG_SSIENR,
                0
            )

        # -------------------------------------------------------------- #
        # 状态2：CONFIG_CTRLR0
        # 配置 Protocol + TMOD + DFS
        # -------------------------------------------------------------- #
        if self._phase == "CONFIG_CTRLR0":
            ctrlr0_value = self._make_ctrlr0(
                protocol=plan["protocol"],
                tmod=plan["tmod"],
                dfs_bits=plan["dfs"],
            )

            self._phase = "CONFIG_SER"

            return self._write_reg(
                self.REG_CTRLR0,
                ctrlr0_value
            )

        # -------------------------------------------------------------- #
        # 状态3：CONFIG_SER
        # 选择从设备
        # -------------------------------------------------------------- #
        if self._phase == "CONFIG_SER":
            self._phase = "CONFIG_BAUDR"

            return self._write_reg(
                self.REG_SER,
                plan["ser"]
            )

        # -------------------------------------------------------------- #
        # 状态4：CONFIG_BAUDR
        # 配置 SPI 分频
        # -------------------------------------------------------------- #
        if self._phase == "CONFIG_BAUDR":
            self._phase = "CONFIG_TXFTLR"

            return self._write_reg(
                self.REG_BAUDR,
                plan["baudr"]
            )

        # -------------------------------------------------------------- #
        # 状态5：CONFIG_TXFTLR
        # 配置 TX FIFO 阈值
        # -------------------------------------------------------------- #
        if self._phase == "CONFIG_TXFTLR":
            self._phase = "ENABLE"

            return self._write_reg(
                self.REG_TXFTLR,
                plan["txftlr"]
            )

        # -------------------------------------------------------------- #
        # 状态6：ENABLE
        #
        # 【关键修改】
        # 所有配置完成之后再 SSIENR=1
        # -------------------------------------------------------------- #
        if self._phase == "ENABLE":
            self._tx_count = 0
            self._wait_count = 0
            self._phase = "WRITE_DR"

            return self._write_reg(
                self.REG_SSIENR,
                1
            )

        # -------------------------------------------------------------- #
        # 状态7：WRITE_DR
        # 连续写入 1~4 笔数据，推动 TX FIFO 和发送 FSM
        # -------------------------------------------------------------- #
        if self._phase == "WRITE_DR":
            pattern_index = (
                self._plan_idx
                + self._repeat_current
                + self._tx_count
            ) % len(self.TX_PATTERNS)

            tx_data = self.TX_PATTERNS[
                pattern_index
            ]

            self._tx_count += 1

            if self._tx_count >= plan["tx_words"]:
                self._phase = "RUN"
                self._wait_count = 0

            return self._write_reg(
                self.REG_DR,
                tx_data
            )

        # -------------------------------------------------------------- #
        # 状态8：RUN
        #
        # 【关键修改】
        # 不再像 random 一样下一拍立刻改寄存器
        # 留足时间让 FSM 真正经历：
        # idle -> assert_ss -> pop_tx -> shift_bit -> ...
        # -------------------------------------------------------------- #
        if self._phase == "RUN":
            self._wait_count += 1

            wait_limit = self._estimate_wait_cycles(
                plan
            )

            # 如果已经快到 max_steps，适当缩短最后一组等待
            if max_steps - step < 1000:
                wait_limit = min(
                    wait_limit,
                    128
                )

            if self._wait_count >= wait_limit:
                self._read_count = 0
                self._phase = "READ_DR"

            return self._idle_action()

        # -------------------------------------------------------------- #
        # 状态9：READ_DR
        # 尝试读取 RX FIFO / DR，顺便覆盖读路径
        # -------------------------------------------------------------- #
        if self._phase == "READ_DR":
            self._read_count += 1

            if self._read_count >= 2:
                self._phase = "FINAL_DISABLE"

            return self._read_reg(
                self.REG_DR
            )

        # -------------------------------------------------------------- #
        # 状态10：FINAL_DISABLE
        #
        # 下一组配置开始前再次明确 SSIENR=0
        # -------------------------------------------------------------- #
        if self._phase == "FINAL_DISABLE":
            self._phase = "EVAL"

            return self._write_reg(
                self.REG_SSIENR,
                0
            )

        # -------------------------------------------------------------- #
        # 状态11：EVAL
        #
        # 【修改14】使用 coverage feedback
        # 有新增 bin -> 当前组合换数据再跑一次
        # 没新增 -> 直接切下一组 plan
        # -------------------------------------------------------------- #
        if self._phase == "EVAL":
            gain = self._coverage_gain(
                coverage_state
            )

            if (
                gain > 0
                and self._repeat_current == 0
            ):
                # 当前组合有效，再利用一次
                self._repeat_current = 1
            else:
                # 当前组合没收益，或者已经复用过一次
                self._advance_plan()

            # 重新记录下一轮起点 coverage
            self._plan_cov_start = (
                coverage_state.copy()
            )

            self._phase = "DISABLE"
            self._wait_count = 0
            self._tx_count = 0
            self._read_count = 0

            # 两轮 transaction 中间插一个 idle
            return self._idle_action()

        # 理论上不会走到这里
        self._phase = "DISABLE"
        return self._idle_action()


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    # 【修改15】smoke test
    # 这里只检查接口、shape、dtype、状态机能否连续运行
    # 不代表真实 RTL coverage 结果
    # ------------------------------------------------------------------ #
    for p in ("hfsm", "random", "greedy"):
        inf = InferenceInterface(
            policy=p
        )

        s = np.zeros(
            inf.total_bins,
            dtype=np.float32
        )

        for step in range(2000):
            a = inf.predict(
                s,
                step,
                50000
            )

            assert a.shape == (
                inf.DIMS,
            )
            assert a.dtype == np.float32
            assert np.all(
                np.isfinite(a)
            )

        print(
            f"[{p}] smoke OK, "
            f"shape={a.shape}, "
            f"dtype={a.dtype}"
        )

    print(
        "InferenceInterface smoke OK"
    )
