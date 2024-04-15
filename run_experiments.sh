# Note: be aware of all parameter settings when running experiments

daily_path='data/bitcoin_daily_epsilon_drawdowns.csv'
hourly_path='data/bitcoin_hourly_epsilon_drawdowns.csv'

daily_start_date='2014-10-07'
daily_end_date='2019-04-06'

# Note: hourly data is only last 730 days
hourly_start_date='2022-04-20'
hourly_end_date='2023-09-20'

win=168
forward=24

# Daily
# python3.9 -m src.bubble_detection.script.run_DL_experiment --drawdowns_path $daily_path  --model_name 'LSTM' --stock 'Bitcoin' --start $daily_start_date --end $daily_end_date --interval '1d' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.1 --win 30 --forward 14 --iteration
# python3.9 -m src.bubble_detection.script.run_DL_experiment --drawdowns_path $daily_path --model_name 'TCN' --stock 'Bitcoin' --start $daily_start_date --end $daily_end_date --interval '1d' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.1 --win 30 --forward 14 --iteration
#
# python3.9 -m src.bubble_detection.script.run_ML_experiment --drawdowns_path $daily_path --model_name 'RF' --stock 'Bitcoin' --start $daily_start_date --end $daily_end_date --interval '1d' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.1 --win 30 --forward 14 --iteration
# python3.9 -m src.bubble_detection.script.run_ML_experiment --drawdowns_path $daily_path --model_name 'SVM' --stock 'Bitcoin' --start $daily_start_date --end $daily_end_date --interval '1d' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.1 --win 30 --forward 14 --iteration
# python3.9 -m src.bubble_detection.script.run_ML_experiment --drawdowns_path $daily_path --model_name 'GBM' --stock 'Bitcoin' --start $daily_start_date --end $daily_end_date --interval '1d' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.1 --win 30 --forward 14 --iteration
# python3.9 -m src.bubble_detection.script.run_ML_experiment --drawdowns_path $daily_path --model_name 'XGB' --stock 'Bitcoin' --start $daily_start_date --end $daily_end_date --interval '1d' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.1 --win 30 --forward 14 --iteration
#
# python3.9 -m src.bubble_detection.script.run_CL_experiment --drawdowns_path $daily_path --stock 'Bitcoin' --start $daily_start_date --end $daily_end_date --interval '1d' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.1 --encoder_type 'TCN' --win 30 --forward 14 --iteration
# python3.9 -m src.bubble_detection.script.run_CL_experiment --drawdowns_path $daily_path --stock 'Bitcoin' --start $daily_start_date --end $daily_end_date --interval '1d' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.1 --encoder_type 'LSTM' --win 30 --forward 14 --iteration

# Hourly
# python3.9 -m src.bubble_detection.script.run_DL_experiment --drawdowns_path $hourly_path  --model_name 'LSTM' --stock 'Bitcoin' --start $hourly_start_date --end $hourly_end_date --interval '60m' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.02 --win $win --forward $forward --iteration
# python3.9 -m src.bubble_detection.script.run_DL_experiment --drawdowns_path $hourly_path --model_name 'TCN' --stock 'Bitcoin' --start $hourly_start_date --end $hourly_end_date --interval '60m' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.02 --win $win --forward $forward --iteration

# python3.9 -m src.bubble_detection.script.run_ML_experiment --drawdowns_path $hourly_path --model_name 'RF' --stock 'Bitcoin' --start $hourly_start_date --end $hourly_end_date --interval '60m' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.02 --win $win --forward $forward --iteration
# python3.9 -m src.bubble_detection.script.run_ML_experiment --drawdowns_path $hourly_path --model_name 'SVM' --stock 'Bitcoin' --start $hourly_start_date --end $hourly_end_date --interval '60m' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.02 --win $win --forward $forward --iteration
# python3.9 -m src.bubble_detection.script.run_ML_experiment --drawdowns_path $hourly_path --model_name 'GBM' --stock 'Bitcoin' --start $hourly_start_date --end $hourly_end_date --interval '60m' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.02 --win $win --forward $forward --iteration
# python3.9 -m src.bubble_detection.script.run_ML_experiment --drawdowns_path $hourly_path --model_name 'XGB' --stock 'Bitcoin' --start $hourly_start_date --end $hourly_end_date --interval '60m' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.02 --win $win --forward $forward --iteration

# python3.9 -m src.bubble_detection.script.run_CL_experiment --drawdowns_path $hourly_path --stock 'Bitcoin' --start $hourly_start_date --end $hourly_end_date --interval '60m' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.02 --encoder_type 'TCN' --win $win --forward $forward --iteration
# python3.9 -m src.bubble_detection.script.run_CL_experiment --drawdowns_path $hourly_path --stock 'Bitcoin' --start $hourly_start_date --end $hourly_end_date --interval '60m' --train_pct 0.8 --peak_frequency 0.5 --drop_percent 0.02 --encoder_type 'LSTM' --win $win --forward $forward --iteration
