# Development Workflow

This document describes the development and documentation deployment workflow for `auto-kappa`.

## Branches

The repository uses the following branches:

- `develop`: main development branch
- `main`: public release branch
- `gh-pages`: published GitHub Pages content

## Basic Workflow

The standard workflow is:

1. Make code or documentation changes on `develop`.
2. Push changes to `develop`.
3. Confirm that the GitHub Actions documentation build succeeds.
4. Merge `develop` into `main`.
5. Push `main`.
6. GitHub Actions automatically deploys the documentation site to `gh-pages`.

## Documentation Deployment

The project website is published with GitHub Pages.

Current publishing configuration:

- Source: `Deploy from a branch`
- Branch: `gh-pages`
- Folder: `/ (root)`

The deployment workflow is defined in:

- `.github/workflows/deploy-pages.yml`

Documentation build dependencies are managed in:

- `docs/requirements.txt`

## Important Rules

- Do not edit the `gh-pages` branch manually during normal operation.
- Update documentation source files on `develop` or `main`.
- The `gh-pages` branch is used only as the published output for GitHub Pages.
- Documentation is automatically rebuilt and deployed when changes are pushed to `main`.

## Development Notes

When updating the documentation system:

- Keep Sphinx-related dependencies in `docs/requirements.txt`.
- Keep GitHub Actions settings in `.github/workflows/deploy-pages.yml`.
- If the documentation build fails in GitHub Actions, check missing Python packages and Sphinx extensions first.

## Recommended Release Flow

A typical release flow is:

1. Develop and test changes on `develop`.
2. Update documentation if needed.
3. Confirm that GitHub Actions succeeds on `develop`.
4. Merge `develop` into `main`.
5. Push `main`.
6. Create a tag on `main` for the release.
7. Confirm that the website has been updated automatically.

## Notes for Contributors

Please keep public-facing user documentation separate from development workflow notes.

- User documentation should remain in `README.md` and related docs pages.
- Internal workflow and maintenance notes should be kept in this file.
