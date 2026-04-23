# Final-Project-Grayson
I know this isn't a lot of work yet, and this write up was last minute. Due to the speech and debate tournament, and frequent tests, I've only just started to keep a journal.\

1. I did extensive research to understand how Statcast calculates Catch Probability (April 16)\
https://www.mlb.com/news/catch-probability-updated-to-include-direction-c232532408\
https://www.mlb.com/news/statcast-introduces-catch-probability-for-2017-c217802340\

Statcast defines Catch Probability as the likelihood that a fielder successfully records an out on a batted ball. Through research, I identified the primary factors influencing this metric:\

Distance Covered (how far the fielder must travel)\
Time to Catch Opportunity (Hang Time)\
Direction of Movement (forward, backward, lateral)\
Proximity to the fence https://www.mlb.com/news/catch-probability-updated-to-account-for-walls-c269814542\


These variables reflect both physical difficulty and reaction constraints for players\

2. Dataset Selection & Constraints April (20-23)\

A Statcast-derived dataset was selected containing batted ball tracking data. I'll aim for one season of data, tracking:\

hc_x, hc_y → ball landing coordinates\
launch_speed, launch_angle\
hit_distance\
fielder_7/8/9 (outfielders)\
events (out vs hit)\


https://www.kaggle.com/datasets/s903124/mlb-statcast-data?








