## Moment-to-Action repository

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branching strategy and collaboration workflow.

If `conda` complains about the relative path (it sometimes does depending on the version), the most bulletproof method is to remove `mobileclip` from the YAML and install it separately.

1.  Remove the `mobileclip` line from environment.yml.
2.  Run:
    ```bash
    conda env create -f environment.yml
    conda activate violence-pipeline
    pip install -e third_party/mobileclip
    ```

However, **Option 1 (Relative Path)** usually works fine in modern Conda/Pip versions and is the cleanest experience.