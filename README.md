# PULSE Python Tools

## Requirements

* Create the environment from withon the repository  
`conda env create -f environment.yml`

* Activate the environment  
Linux: `source activate pulsept`  
Windows: `activate pulsept`

* Set the environment variables, e.g.
```
export PULSE_DATA_DIR=/netshares/ibme_biomedia/Projects_1/pulse2
export LOCAL_DATA_DIR=<some-local-directory>
```


## Anatomical Annotations

* Manually enter the scan_id and the label destination folder in the _run_anatomical_annotations.py_ file.
* Execute the script with  
`python run_anatomical_annotations.py`

