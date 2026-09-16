# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Raise an explicit `ValueError` when `pkg_payload.zip` or `data_payload.zip` exceeds 5 MB instead of silently dropping the payload.

### Changed

- Set payload size limit to 5 MB during project and local data packaging to prevent oversized inline kernel payloads.

## [0.1.0] - 2026-09-16

### Added

- Initial release of `kagglex` CLI and Python toolkit.
- Automatic packaging of local code and dependencies for Kaggle kernels.
- Remote execution on Kaggle GPUs and CPUs.
- Live output streaming and artifact synchronization.
- Pre-flight workspace validation and `.kaggleignore` filtering.
- Dataset upload and staging management.
