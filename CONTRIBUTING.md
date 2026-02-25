# Contributing to Moment-to-Action

Thank you for your interest in contributing! This document describes our branching strategy and workflow so that multiple contributors can collaborate effectively.

## Branch Structure

```
main
 └── dev
      ├── <username>/dev
      ├── <username>/feature-<short-description>
      ├── <username>/bugfix-<short-description>
      └── <username>/experiment-<short-description>
```

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code only. Direct pushes are not allowed. |
| `dev` | Shared integration branch. Features are merged here first before `main`. |
| `<username>/dev` | Your personal long-running development branch. |
| `<username>/feature-<desc>` | A focused addition of new functionality. |
| `<username>/bugfix-<desc>` | A targeted fix for a specific bug. |
| `<username>/experiment-<desc>` | Exploratory ML experiments that may or may not be merged. |

## Branch Naming Rules

- Use your GitHub username as the prefix (e.g. `soma/`, `kausar/`).
- Use lowercase letters and hyphens; avoid underscores and spaces.
- Keep descriptions short and descriptive (e.g. `soma/feature-yolo-pose`, `kausar/experiment-mobilenet-v3`).

## Workflow

1. **Create your branch** from `dev` (not from `main`):
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b <username>/<type>-<short-description>
   ```

2. **Make your changes**, committing frequently with clear messages:
   ```bash
   git add .
   git commit -m "feat: add YOLO-pose inference pipeline"
   ```

3. **Keep your branch up to date** with `dev`:
   ```bash
   git fetch origin
   git rebase origin/dev
   ```

4. **Open a Pull Request** from your branch into `dev`.
   - Write a clear title and description.
   - Link any related issues.
   - Request at least one reviewer before merging.

5. **Merging to `main`** is done periodically from `dev` once the integration branch is stable and reviewed.

## Commit Message Style

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <short summary>
```

Common types: `feat`, `fix`, `experiment`, `docs`, `refactor`, `chore`.

Examples:
- `feat: add MobileNet violence classifier`
- `experiment: benchmark MoViNet on Rubik Pi`
- `fix: correct frame sampling rate in movinet pipeline`
- `docs: update environment setup instructions`
