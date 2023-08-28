## Data Preprocessing for EMI

This dir contains scripts to process project specific data into a format ready to perform EMI. I finished integrating `DLATKDataGetter` in the src dir, which takes in the message table, associated feature table, outcome table and outcome fields to create a datadictionary. 

### Files

`utils.py`: Has functions to add the `src` dir to the path. 

`ds4ud_anilsson.py`: Use the EMA data processed August into data dictionary. Currently looking at average drinking (anscombe) per wave, average affect, average energy and average wellbeing.

### Commands:

```python ds4ud_anilsson.py --save_filepath /data/avirinchipur/EMI/datadicts/drinking_ans_avg.pkl```