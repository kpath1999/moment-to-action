Audio -> screaming/classes (yamnet) -- always running
500 classes -- LLM as a layer to find instances of violence
If this has happened for quite a while
Call to function
Load frames (yolo)
Call to function
Load video (mobileclip)

```
source /Users/kausar/miniconda3/bin/activate violence-pipeline && export PATH="/Users/kausar/miniconda3/envs/violence-pipeline/bin:$PATH" && hash -r && which python
```

### Alternative: The "Two-Step" Approach (Most Reliable)

If `conda` complains about the relative path (it sometimes does depending on the version), the most bulletproof method is to remove `mobileclip` from the YAML and install it separately.

1.  Remove the `mobileclip` line from [environment.yml](http://_vscodecontentref_/2).
2.  Run:
    ```bash
    conda env create -f environment.yml
    conda activate violence-pipeline
    pip install -e third_party/mobileclip
    ```

However, **Option 1 (Relative Path)** usually works fine in modern Conda/Pip versions and is the cleanest experience.