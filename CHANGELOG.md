# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.2.1](https://github.com/pando85/battery-optimizer/tree/v0.2.1) - 2026-10-25

### Fixed

- extract: Exit non-zero when a run records coverage gaps ([44eb411](https://github.com/pando85/battery-optimizer/commit/44eb411ad341c9ceb9a355ec09ee492bc00cfe13))

## [v0.2.0](https://github.com/pando85/battery-optimizer/tree/v0.2.0) - 2026-10-25

### Added

- service: Add the resident shadow service ([a1d52fe](https://github.com/pando85/battery-optimizer/commit/a1d52fef8b571d6515b9b339fde4bbc3f499aeb5))

### CI

- Lint the PR title with commitlint ([173e9ee](https://github.com/pando85/battery-optimizer/commit/173e9ee53f53ed22cedca77dc38279830888775c))
- Attach only the wheel and sdist to GitHub releases ([f6d935e](https://github.com/pando85/battery-optimizer/commit/f6d935e414cd2c51656153afa951c7693f51aa45))

## [v0.1.0](https://github.com/pando85/battery-optimizer/tree/v0.1.0) - 2026-10-25

### Added

- manifest: Accept a build-time commit identity when git is unavailable ([db1993a](https://github.com/pando85/battery-optimizer/commit/db1993abbd3b23fbbfeb33175ebf3f05e9ad1954))

### Documentation

- battery: Decide production runtime and independent repo ownership (pando85/homelab#4653) ([97fdf17](https://github.com/pando85/battery-optimizer/commit/97fdf1734f9b74704ab07e9ea0154f66e530afca))
- battery: Document Omnibattery HA control contract and integration gates (pando85/homelab#4654) ([b184ed2](https://github.com/pando85/battery-optimizer/commit/b184ed2ca6227c5428aac648a8216aaa9b6862b2))
- Propose phased home battery economic optimizer ADR (pando85/homelab#4651) ([3e01138](https://github.com/pando85/battery-optimizer/commit/3e01138231f41d85a689502d0c2b53d77934c5b4))
- ADR-0002 appliance-aware load forecasting with HA telemetry and NILM (pando85/homelab#4652) ([5a2cb0d](https://github.com/pando85/battery-optimizer/commit/5a2cb0d5c2e8961a4f5530d75323de73d04e47c5))
- Define reduced-scope energy baseline ADR ([bccfb0b](https://github.com/pando85/battery-optimizer/commit/bccfb0b25fe0dc737a7f2b731ada93e3f6104a20))
- Record production execution path for energy baseline ([e470103](https://github.com/pando85/battery-optimizer/commit/e4701037635bc2efd6dda2fa7de71409a921f143))
- Record conditional acceptance of the four-policy baseline ([b6a01e8](https://github.com/pando85/battery-optimizer/commit/b6a01e85617011bb3705274eff46e318bdd6d3b2))
- Record owner-supplied battery and price facts ([43a696d](https://github.com/pando85/battery-optimizer/commit/43a696d6e2d3f590bab7a2e66726f4c757d25df3))
- Fill rebuild placeholders with the 0.80 ladder results ([a379e3f](https://github.com/pando85/battery-optimizer/commit/a379e3fa5440968950f9eba37554450465a68b5c))
- Record the bounded pre-migration milestone ([f9033ba](https://github.com/pando85/battery-optimizer/commit/f9033ba5f49971344f2ce369c652fb3605c31357))
- Resolve contradictory milestone claims and disclose provenance ([d765013](https://github.com/pando85/battery-optimizer/commit/d7650139555eb08367d34e605c00d5ec3ddcfa9a))
- Record pinned re-runs and the causal five-rung ladder ([7d64b5b](https://github.com/pando85/battery-optimizer/commit/7d64b5be9bacd5409db4732e37b4f5c3d5e4b942))
- Record the independent review of the pinned evidence ([0dd35be](https://github.com/pando85/battery-optimizer/commit/0dd35be8ff0506e61a9002d51d5377c7b46ecc06))
- Fix links and paths in the moved ADRs and planning notes ([e70d2f2](https://github.com/pando85/battery-optimizer/commit/e70d2f2272ea36a6bea785ae0ce871ccce6a110e))
- Record production sequencing as phases A to D ([6bab29c](https://github.com/pando85/battery-optimizer/commit/6bab29cd74e8ba3e87b90429116224ba0bd14279))
- Add provenance of the homelab extraction and a docs index ([2b407c1](https://github.com/pando85/battery-optimizer/commit/2b407c17ade984afcd02c65fdf158a4caf141516))
- Add badges, container usage and environment variables to the README ([fc77027](https://github.com/pando85/battery-optimizer/commit/fc77027e3596db39679f2055eab6881bc223f67d))
- Add AGENTS.md and CLAUDE.md ([01d3b50](https://github.com/pando85/battery-optimizer/commit/01d3b50a582fd6196427a1f8b4b5c0f671292aba))

### Build

- docker: Add multi-stage container image ([3c58df4](https://github.com/pando85/battery-optimizer/commit/3c58df4ae5664990beb6b3596fd1304c7b1373c8))
- Target Python 3.14 ([d9eebfd](https://github.com/pando85/battery-optimizer/commit/d9eebfd618bb5a99fb8fcfd8eba93b9f290054e7))
- Add package metadata and a tools dependency group ([b98a064](https://github.com/pando85/battery-optimizer/commit/b98a064dbf05b3d52f11e4da860df3f66ddbf0e4))
- Add Makefile ([68d55b0](https://github.com/pando85/battery-optimizer/commit/68d55b030fcb4ef607a47a5bfb2aba043c78a3b5))
- Add changelog generation and release script ([9597c04](https://github.com/pando85/battery-optimizer/commit/9597c04c741476d9fb2d71c9c7fc6f05749df260))
- Add pre-commit hooks and commit linting ([51d9b24](https://github.com/pando85/battery-optimizer/commit/51d9b24c59ea61a09660185be51d0ba21eb735ea))
- Create CHANGELOG.md on the first release ([89d072b](https://github.com/pando85/battery-optimizer/commit/89d072b36ad062c391da8464252004554a3417f1))
- Normalize trailing newlines of the generated changelog ([5a3dfa9](https://github.com/pando85/battery-optimizer/commit/5a3dfa94acfa90514ef3fc562acfbfea2302f08f))
- Replace the Makefile with a justfile ([639afcd](https://github.com/pando85/battery-optimizer/commit/639afcdb04679a4a59d810c8cee37ead6d071393))

### CI

- Add lint and test workflow ([97efa25](https://github.com/pando85/battery-optimizer/commit/97efa255e8a5914b9896d756933371398f8096de))
- Add pre-commit workflow ([37be66a](https://github.com/pando85/battery-optimizer/commit/37be66a50f1d9545d8babd04401f23a0328b920c))
- Add image build, publish and release workflow ([dc4c660](https://github.com/pando85/battery-optimizer/commit/dc4c6607b77b33d7e21b94844cbec015d4ce9b91))
- Add auto-tag workflow ([a5f424b](https://github.com/pando85/battery-optimizer/commit/a5f424b0006fe8317e95292f6aad45600cd71d88))
- Feed commit messages to commitlint through COMMIT_EDITMSG ([be309a8](https://github.com/pando85/battery-optimizer/commit/be309a8bd9e91b66119dd73c613d38c22aca2292))

### Chore

- Add Renovate config ([0fc61c7](https://github.com/pando85/battery-optimizer/commit/0fc61c758c7f39853a483ee546f307595b84bbea))
- Drop the redundant Renovate regex manager ([47c3066](https://github.com/pando85/battery-optimizer/commit/47c30667aa112ce5c52677892f59c76060a50bed))
- Add GitHub community files ([7ae38cf](https://github.com/pando85/battery-optimizer/commit/7ae38cfac94fc764ae27d15d0c1eeb3f45cd3865))
- Add direnv example and ignore .envrc ([d1e9bd3](https://github.com/pando85/battery-optimizer/commit/d1e9bd37709533f9eaf095e30f8e9ed4aa049b23))

### Refactor

- Rename package to battery_optimizer ([3b5a93a](https://github.com/pando85/battery-optimizer/commit/3b5a93ab09280eba18a75f6f7533750feb99fc57))

### Testing

- Resolve git-tree refusal fixture against the repository root ([67e99ef](https://github.com/pando85/battery-optimizer/commit/67e99ef2f9953dc7bec0c4692b9ead8f8e17143a))
- Compare golden cost totals with a 1e-12 relative tolerance ([2e3504c](https://github.com/pando85/battery-optimizer/commit/2e3504cf2188e46b6669710d65c46c63688d583c))

### Scripts

- Add energy-baseline package with Prometheus and HA extraction ([0b9e726](https://github.com/pando85/battery-optimizer/commit/0b9e72647e1c69692af42aa4113c19064f86458b))
- Complete energy-baseline benchmark and correct build facts ([fc6375d](https://github.com/pando85/battery-optimizer/commit/fc6375d1eef108e6a16cc060ef80a7d9b3a74d99))
- Deduplicate HA re-extraction and record Phase 1 decisions ([5b735a5](https://github.com/pando85/battery-optimizer/commit/5b735a5b03a18d7cb5b5a17a040e067500e738a2))
- Add day-ahead policy on a shared scoring basis ([0ce3447](https://github.com/pando85/battery-optimizer/commit/0ce3447e1d699a40545a76fa5c19c0660d4e6bbb))
- Add two-publication rolling-horizon experiment ([3eb41a9](https://github.com/pando85/battery-optimizer/commit/3eb41a9f7ab1c7d29c635b991180a641711ae233))
- Promote rolling horizon to a fifth policy at RTE 0.80 ([bda3d7d](https://github.com/pando85/battery-optimizer/commit/bda3d7d522be6a14da777a97faf4bc6d400c9e1e))
- Correct false forecast-basis comparability claim ([395a408](https://github.com/pando85/battery-optimizer/commit/395a408f968e568fb94e440221f783d8d8b4443a))
- Add artifact-integrity manifest module and report wiring ([0bfec3b](https://github.com/pando85/battery-optimizer/commit/0bfec3ba438fe711d8b73ce9e424fefa36cfa920))
- Align rolling-horizon load forecast with the day-ahead basis ([d7f8b06](https://github.com/pando85/battery-optimizer/commit/d7f8b062f5d85e7c4b280e191289e7c12d8fd3de))
- Add sensitivity sweeps, replay runtime, and validation runner ([87afbb6](https://github.com/pando85/battery-optimizer/commit/87afbb6356d0794ca10b77d81a474ebd4ce39a92))
- Preserve plan sequence across restart in the replay runtime ([173fb3f](https://github.com/pando85/battery-optimizer/commit/173fb3ffb15881432f6687e08cb7b04a5ebeaade))
- Correct docstring overpromises and derive slots per day ([428b019](https://github.com/pando85/battery-optimizer/commit/428b0190bec45878cdf076268abf9e0393d231fa))
- Pin code provenance in both experiment runners ([f57c27b](https://github.com/pando85/battery-optimizer/commit/f57c27ba4c4de83ffce3704e891541642b111b1e))
