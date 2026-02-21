# Guide to run the script

Thank you for contributing to this repository. This project is optimized for **source-code–only collaboration**, primarily for experiments and development on the Thundercomm Rubik Pi platform.

Please read this document carefully before adding or modifying code.

Currently, only the script yamnet_scream_tune.py is used to run YAMNet + CLIP

Please use the following instruction to run the code:

```bash
python3 -m venv <venv_name>
source <venv_name>/bin/activate
pip install -r requirements.txt
```

#If using one of the .wav files
```bash
python3 yamnet_scream_tune.py fighting_audio/<name of audio file>.wav
```

#If using the microphone
```bash
python3 yamnet_scream_tune.py
```

---
