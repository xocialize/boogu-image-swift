// FlowMatchEulerDiscreteScheduler with Boogu static-v1 time-shift — Swift mirror of
// boogu_image_mlx/scheduler.py. The schedule is pure scalar math (computed in Double,
// stored as Float, matching numpy-float64-then-astype); only the Euler `step` touches
// the latent. Base config: do_shift, static v1 time-shift, seq_len 4096.

import Foundation
import MLX

public final class FlowMatchEulerDiscreteScheduler {
    let config: BooguSchedulerConfig

    /// The shifted timesteps for the current run, plus the trailing 1.0 sentinel.
    public private(set) var timesteps: [Float] = []
    private var paddedTimesteps: [Float] = []  // timesteps + [1.0]

    public init(_ config: BooguSchedulerConfig) {
        self.config = config
    }

    public convenience init(directory: URL) throws {
        self.init(try BooguSchedulerConfig.load(
            directory.appendingPathComponent("scheduler_config.json")))
    }

    /// y = m*x + b through (x1,y1)-(x2,y2); evaluated at `x`.
    private static func linFunction(
        _ x: Double, x1: Double = 256, y1: Double, x2: Double = 4096, y2: Double
    ) -> Double {
        let m = (y2 - y1) / (x2 - x1)
        let b = y1 - m * x1
        return m * x + b
    }

    private static func timeShiftV1(_ t: [Double], mu: Double, sigma: Double = 1.0) -> [Double] {
        let eps = 1e-8
        let num = Foundation.exp(mu)
        return t.map { ti in
            let t1 = min(max(1.0 - ti, eps), 1.0 - eps)
            let denom = num + Foundation.pow(1.0 / t1 - 1.0, sigma)
            return 1.0 - num / denom
        }
    }

    private static func timeShiftV2(_ t: [Double], m: Double) -> [Double] {
        t.map { ti in ti / (m - m * ti + ti) }
    }

    /// Mirror of scheduler.set_timesteps: linspace(0,1,steps+1)[:-1] + time-shift.
    @discardableResult
    public func setTimesteps(_ numInferenceSteps: Int, numTokens: Int? = nil) -> [Float] {
        let n = numInferenceSteps
        var t: [Double] = (0..<n).map { Double($0) / Double(n) }  // linspace(0,1,n+1)[:-1]

        if config.doShift {
            let v = config.timeShiftVersion
            let v2Scale = Double(config.timeShiftV2HalfScalingFactor) * 2
            if config.dynamicTimeShift {
                if v == "v1", let tokens = numTokens {
                    let reduced = Double(max(1, tokens / 4))
                    let mu = Self.linFunction(
                        reduced, y1: Double(config.baseShift), y2: Double(config.maxShift))
                    t = Self.timeShiftV1(t, mu: mu)
                } else if v == "v2", let tokens = numTokens {
                    t = Self.timeShiftV2(t, m: Double(tokens).squareRoot() / v2Scale)
                }
            } else if let seqLen = config.seqLen {
                if v == "v1" {
                    let mu = Self.linFunction(
                        Double(seqLen), y1: Double(config.baseShift), y2: Double(config.maxShift))
                    t = Self.timeShiftV1(t, mu: mu)
                } else if v == "v2" {
                    t = Self.timeShiftV2(t, m: Double(seqLen).squareRoot() / v2Scale)
                }
            }
        }

        timesteps = t.map { Float($0) }
        paddedTimesteps = timesteps + [1.0]
        return timesteps
    }

    /// Euler flow step. `stepIndex` indexes `timesteps` explicitly (no internal state).
    public func step(_ modelOutput: MLXArray, stepIndex: Int, sample: MLXArray) -> MLXArray {
        let t = paddedTimesteps[stepIndex]
        let tNext = paddedTimesteps[stepIndex + 1]
        return sample + (tNext - t) * modelOutput
    }
}
