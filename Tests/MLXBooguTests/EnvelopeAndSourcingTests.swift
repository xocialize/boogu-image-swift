// MLXBoogu offline gates beside CAN: MAT (auto-materialization) for every published tier, the tier
// repo table, the output-size envelope the footprints cover (AB-T-0204), and the snapshot quant check.
// No network, no weights, no Metal kernels.
import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXBoogu

final class EnvelopeAndSourcingTests: XCTestCase {

    /// A snapshot + conditioner pair that satisfies the sources, with an optional quant_config.json.
    private func satisfiedDirs(bits: Int? = nil) throws -> (snapshot: URL, qwen: URL, cleanup: () -> Void) {
        let fm = FileManager.default
        let base = fm.temporaryDirectory.appending(path: "boogu-mat-\(UUID().uuidString)")
        let snapshot = base.appending(path: "snapshot"), qwen = base.appending(path: "qwen")
        for sub in ["transformer", "vae", "scheduler"] {
            try fm.createDirectory(at: snapshot.appending(path: sub), withIntermediateDirectories: true)
        }
        try fm.createDirectory(at: qwen, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: qwen.appending(path: "config.json"))
        if let bits {
            try Data(#"{"group_size": 32, "bits": \#(bits)}"#.utf8)
                .write(to: snapshot.appending(path: "transformer/quant_config.json"))
        }
        return (snapshot, qwen, { try? fm.removeItem(at: base) })
    }

    // MARK: MAT

    func testMATGatePassesEveryPublishedTier() throws {
        let (snapshot, qwen, cleanup) = try satisfiedDirs()
        defer { cleanup() }
        let tiers: [BooguImageConfiguration] = [
            .base(quant: .int4), .base(quant: .int8), .base(quant: .bf16),
            .turbo(quant: .int8), .turbo(quant: .bf16),
            .edit(quant: .int4), .edit(quant: .int8), .edit(quant: .bf16),
        ]
        for fresh in tiers {
            var satisfied = fresh
            satisfied.snapshotPath = snapshot.path
            satisfied.qwenPath = qwen.path
            let report = MaterializationConformance.check(freshConfiguration: fresh, satisfiedConfiguration: satisfied)
            XCTAssertTrue(report.passed, "\(fresh.repo)\n\(report.summary)")
        }
    }

    func testSourcesAreTheTierAndTheSharedConditioner() {
        let sources = BooguImageConfiguration.edit(quant: .bf16).weightSources
        XCTAssertEqual(sources.map(\.role), ["snapshot", "conditioner"])
        XCTAssertEqual(sources.map(\.repo), ["mlx-community/Boogu-Image-0.1-Edit-bf16", "Qwen/Qwen3-VL-8B-Instruct"])
    }

    func testAnExplicitPathSatisfiesOnlyItsOwnSource() throws {
        let (snapshot, _, cleanup) = try satisfiedDirs()
        defer { cleanup() }
        var config = BooguImageConfiguration.base(quant: .int4)
        config.snapshotPath = snapshot.path
        let store = FileManager.default.temporaryDirectory.appending(path: "boogu-empty-store-\(UUID().uuidString)")
        XCTAssertEqual(config.missingWeightSources(storeRoot: store).map(\.role), ["conditioner"])
    }

    func testTierRepos() {
        XCTAssertEqual(BooguImageConfiguration.repo(.base, .int4), "mlx-community/Boogu-Image-0.1-Base-4bit")
        XCTAssertEqual(BooguImageConfiguration.repo(.base, .int8), "mlx-community/Boogu-Image-0.1-Base-8bit")
        XCTAssertEqual(BooguImageConfiguration.repo(.turbo, .int8), "mlx-community/Boogu-Image-0.1-Turbo-8bit")
        XCTAssertEqual(BooguImageConfiguration.repo(.edit, .bf16), "mlx-community/Boogu-Image-0.1-Edit-bf16")
        XCTAssertNil(BooguImageConfiguration.repo(.turbo, .int4), "Turbo is quant-sensitive at int4; not published")
        XCTAssertNil(BooguImageConfiguration.repo(.base, .fp16))
        let turbo = BooguImageConfiguration.turbo()
        XCTAssertEqual(turbo.quant, .int8)
        XCTAssertEqual(turbo.defaultSteps, 4)
        XCTAssertEqual(turbo.defaultGuidance, 1.0)
    }

    // MARK: Envelope (AB-T-0204)

    func testOutputSizeFloorsToTheGrid() throws {
        let (w, h) = try BooguImagePackage.outputSize(1000, 770)
        XCTAssertEqual(w, 992)
        XCTAssertEqual(h, 768)
    }

    func testOutputSizeAcceptsTheCapAndRefusesAbove() throws {
        let side = Int(Double(BooguImagePackage.maxOutputPixels).squareRoot())
        XCTAssertNoThrow(try BooguImagePackage.outputSize(side, side))
        XCTAssertThrowsError(try BooguImagePackage.outputSize(side + 16, side + 16))
        // The cap is on area: a non-square size of no more area is admitted.
        let wide = min(BooguImagePackage.maxOutputSide, side * 5 / 4 / 16 * 16)
        XCTAssertNoThrow(try BooguImagePackage.outputSize(wide, BooguImagePackage.maxOutputPixels / wide / 16 * 16))
    }

    func testOutputSizeRefusesSidesOutsideTheRange() {
        XCTAssertThrowsError(try BooguImagePackage.outputSize(496, 1024))
        XCTAssertThrowsError(try BooguImagePackage.outputSize(2064, 768))
        XCTAssertNoThrow(try BooguImagePackage.outputSize(512, 512))
        XCTAssertNoThrow(try BooguImagePackage.outputSize(2048, 1152))  // 16:9 at the T2I area cap
    }

    func testEditHasItsOwnSmallerCap() {
        XCTAssertNoThrow(try BooguImagePackage.outputSize(1024, 1024, maxPixels: BooguImagePackage.maxEditOutputPixels))
        XCTAssertThrowsError(try BooguImagePackage.outputSize(1280, 1024, maxPixels: BooguImagePackage.maxEditOutputPixels))
        XCTAssertNoThrow(try BooguImagePackage.outputSize(1280, 1024))
    }

    func testWorkloadIsTheFlooredOutputArea() {
        let config = BooguImageConfiguration.base()
        XCTAssertEqual(config.workloadUnits(for: T2IRequest(prompt: "p", width: 1000, height: 770)), 992 * 768)
        XCTAssertEqual(config.workloadUnits(for: T2IRequest(prompt: "p")), 1024 * 1024)
        let png = Image(format: .png, data: Data(), width: 8, height: 8)
        XCTAssertEqual(BooguImageConfiguration.edit().workloadUnits(for: IEditRequest(images: [png], prompt: "p")),
                       768 * 768)
    }

    // MARK: Footprint (FIT gate, AB-T-0204)

    func testFootprintIsWellFormedAndEnforceable() {
        let report = FootprintConformance.check(
            manifest: BooguImagePackage.manifest, configuration: BooguImageConfiguration.base())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testTheLineCoversEveryMeasuredPoint() throws {
        // (output pixels, measured activation GB) from AB-A-0224's matrix, every quant.
        let measured: [(Double, Double)] = [(512 * 512, 16.57), (768 * 768, 16.57), (1024 * 1024, 16.57),
                                            (1024 * 1024, 18.80), (1536 * 1536, 23.34)]
        for footprint in BooguImagePackage.manifest.requirements.footprints {
            let line = try XCTUnwrap(footprint.activationScaling)
            XCTAssertEqual(line.measuredCeiling, Double(BooguImagePackage.maxOutputPixels))
            for (pixels, gb) in measured {
                let reserve = max(Double(footprint.peakActivationBytes), Double(line.projectedBytes(at: pixels)))
                XCTAssertGreaterThanOrEqual(reserve, gb * 1e9, "\(footprint.quant) at \(Int(pixels)) px")
            }
        }
    }

    // MARK: Snapshot quant

    func testSnapshotQuantReadsQuantConfig() throws {
        for (bits, quant) in [(nil, Quant.bf16), (4, .int4), (8, .int8)] as [(Int?, Quant)] {
            let (snapshot, _, cleanup) = try satisfiedDirs(bits: bits)
            defer { cleanup() }
            XCTAssertEqual(try BooguImagePackage.snapshotQuant(
                transformerDir: snapshot.appending(path: "transformer")), quant)
        }
    }

    func testLoadRefusesAQuantThatDoesNotMatchTheSnapshot() async throws {
        let (snapshot, qwen, cleanup) = try satisfiedDirs(bits: 4)
        defer { cleanup() }
        let package = BooguImagePackage(configuration: BooguImageConfiguration(
            snapshotPath: snapshot.path, qwenPath: qwen.path, quant: .bf16))
        do {
            try await package.load()
            XCTFail("load() accepted a bf16 configuration over an int4 snapshot")
        } catch BooguImagePackageError.quantMismatch(let declared, let found) {
            XCTAssertEqual(declared, .bf16)
            XCTAssertEqual(found, .int4)
        }
    }
}
