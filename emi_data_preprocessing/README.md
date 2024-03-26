## Data Preprocessing for EMI

This dir contains scripts to process project specific data into a format ready to perform EMI. I finished integrating `DLATKDataGetter` in the src dir, which takes in the message table, associated feature table, outcome table and outcome fields to create a datadictionary. 

### Files

`utils.py`: Has functions to add the `src` dir to the path. 

`ds4ud_anilsson.py`: Use the EMA data processed August into data dictionary. Currently looking at average drinking (anscombe) per wave, average affect, average energy and average wellbeing.

### Commands:

```python ds4ud_anilsson.py --save_filepath /data/avirinchipur/EMI/datadicts/drinking_ans_avg.pkl```

```python voicemails_ema.py --outcome_field PCL11_ans_avg --save_filepath /data/avirinchipur/EMI/datadicts/rift_PCL11_ans_avg.pkl --test_ratio 0.15 --val_ratio 0.10```

```python voicemails_ema.py --outcome_field PCL11_minmax_ans_avg --save_filepath /data/avirinchipur/EMI/datadicts/rift_PCL11_minmax_ans_avg.pkl --test_ratio 0.15 --val_ratio 0.10```

```python voicemails_ema.py --outcome_field PCL11_minmax_ans_avg --save_filepath /data/avirinchipur/EMI/datadicts/rift_PCL11_minmax_ans_avg_roba32.pkl --test_ratio 0.15 --val_ratio 0.10```

```python voicemails_ema.py --outcome_field PCL11_minmax_ans_avg --save_filepath /data/avirinchipur/EMI/datadicts/rift_PCL11_minmax_ans_avg_roba64.pkl --test_ratio 0.15 --val_ratio 0.10```

```python voicemails_ema.py --outcome_field PCL11_minmax_ans_avg --save_filepath /data/avirinchipur/EMI/datadicts/rift_PCL11_minmax_ans_avg_roba128.pkl --test_ratio 0.15 --val_ratio 0.10```

```python voicemails_ema.py --outcome_field PCL11_minmax_ans_avg --save_filepath /data/avirinchipur/EMI/datadicts/rift_PCL11_minmax_ans_avg_roba32_rangel3u14.pkl --test_ratio 0.15 --val_ratio 0.10```


```python voicemails_ema.py --outcome_field PCL11_minmax_ans_avg --save_filepath /data/avirinchipur/EMI/datadicts/voicemails_PCL11_minmax_ans_avg_roba64_rangel3u14_10folds.pkl --test_ratio 0.15 --val_ratio 0.0 --num_folds 10```

```python voicemails_ema.py --outcome_field PCL11_minmax_ans_avg --save_filepath /data/avirinchipur/EMI/datadicts/voicemails_PCL11_minmax_ans_avg_roba128_rangel3u14_10folds.pkl --test_ratio 0.15 --val_ratio 0.0 --num_folds 10```