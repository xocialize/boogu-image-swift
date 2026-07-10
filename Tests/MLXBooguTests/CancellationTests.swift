// CancellationTests.swift — Boogu-Image (Lumina2/NextDiT + Qwen3-VL conditioner) through
// the engine's CAN gate (offline, no MLX kernels, no weights). CAN-1/2 drive the real run()
// pre-cancelled: the entry checkpoint (`try Task.checkCancellation()` as the FIRST act of
// run(), before notLoaded validation) fires before weights are touched, so a stub
// configuration suffices. CAN-3 is the document of record for the checkpoint cadence:
//   - post-encode seams — `try Task.checkCancellation()` in the wrapper's t2i and edit arms
//     right after the Qwen3-VL conditioner is encoded + evicted (BooguImagePackage.run);
//   - denoise/step — `if Task.isCancelled { break }` at the top of the denoise loop in
//     BooguImageGenerator.denoise (Sources/BooguImage/Pipeline.swift; non-throwing core —
//     sanctioned break shape), shared by t2i and edit;
//   - pre-decode seam — a cancelled task skips the monolithic VAE decode (ONE MLX eval, no
//     chunk loop, so no per-chunk decode cadence is claimed);
//   - the wrapper's post-generate `try Task.checkCancellation()` rethrows the
//     CancellationError UNCHANGED (the only catch blocks live in the BooguGate CLI, off the
//     run() path — nothing to launder, CAN-2).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXBoogu

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config (empty paths — never touched: the pre-cancelled run() throws at the
        // entry checkpoint before load-state validation); construction is cheap (C13).
        let package = BooguImagePackage(configuration: BooguImageConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // 20 GB declared peak activation implies long runs — no sub-second exemption.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: BooguImagePackage.manifest))
        let report = CancellationConformance.checkCadence(
            manifest: BooguImagePackage.manifest,
            posture: .cadence([
                // Per-denoise-step Task.isCancelled break in BooguImageGenerator.denoise
                // (shared by textToImage and imageEdit); post-encode-evict + pre-decode
                // seams bracket it (single forwards — seams, not recurring units).
                .init(phase: .denoise, unit: .step),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
