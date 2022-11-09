import time

def trans_time(time_str):
    res = time_str.split(':')
    assert len(res) == 4
    cur_sec = 0
    time_cof = [3600*24, 3600, 60, 1]
    for i in range(len(res)):
        mid_res = int(res[i]) * time_cof[i]
        cur_sec = cur_sec +  mid_res
    return cur_sec

# Full_slurp
# ed_time = "02:05:01:26"
# st_time = "00:20:10:00"

# tts_slurp 22000+
# ed_time = "01:00:55:58"
# st_time = "00:14:58:32"

# slu full slue
# ed_time = "17:01:25:46"
# st_time = "07:15:58:04"

# slu random slue
ed_time = "00:19:32:52"
st_time = "00:14:50:54"

# slu our slue
# ed_time = "00:08:10:13"
# st_time = "00:01:51:38"

total_time = trans_time(ed_time) - trans_time(st_time)
total_hour = total_time * 1.0 / 3600
print(total_hour)
