# Source - https://stackoverflow.com/a/42544963
# Posted by raphinesse, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-17, License - CC BY-SA 4.0

git rev-list --objects --all --missing=print |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  sed -n 's/^blob //p' |
  sort --numeric-sort --key=2 |
  cut -c 1-12,41- |
  $(command -v gnumfmt || echo numfmt) --field=2 --to=iec-i --suffix=B --padding=7 --round=nearest
