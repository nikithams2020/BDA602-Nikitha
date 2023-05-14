
Data Problems In the two tables there were many columns with same name it created some confusions Eg: Fly_Out and flyout Force_Out and force_out Tried to leave some the columns as it could cheat We also see that that home_streak and away_streak columns tend to cheat the prediction.

Feature Selection Tables – Pitcher Counts and BoxScore Response- Home Team Wins WHY these Tables? Pitcher Counts A pitcher's main objective is to prevent the opposing team from scoring runs, and they do this by throwing pitches that are difficult for the batter to hit or by inducing ground balls or fly balls that can be easily fielded by the defensive team. Box Score Game Statistics and also player’s performances and Team’s performances

Analyzing Baseball Dataset Initial Features and Tables Tables – Pitcher Counts and BoxScore Response- Home Team Wins Features w.r.t Home team: Strikeout Walk OutsPlayed Pitches Thrown Pitcher atBat toBase Single Double Groundout DaysSinceLastPitch Features w.r.t Away team: Strikeout Walk OutsPlayed Pitches Thrown Pitcher atBat toBase Single Double Groundout DaysSinceLastPitch Features w.r.t Boxscore: home_runs home_hits away_runs away_hits away_errors home_errors Own Features: home_runrate [home_runs/home_hits] Away_runrate [ away_runs/away_hits] Home_Error_Rate/Home_hits Away_Error_Rate/Away_hits

 Modelling and Reducing Features
 Models Implemented:
Random Forest                  - 81.10%
Logistic Regression			-53.53.0%
Support Vector Machine  -52.71%
Neural Network Classifier. -53.56%

 
game         | ['H_atBat', 'away_team_id', 'home_team_id', 'A_rolling_pitches_thrown', 'H_rolling_pitches_thrown', 'H_pitcher', 'A_pitcher']
game         | 81.10657026092424%
game         | 53.53662370323797%
game         | 52.71927066960076%
game         | 53.56806035837786%
game         |               precision    recall  f1-score   support
game         | 
game         |            0       0.79      0.80      0.80      1471
game         |            1       0.83      0.82      0.82      1710    | 
game         |     accuracy                           0.81      3181
game         |    macro avg       0.81      0.81      0.81      3181
game         | weighted avg       0.81      0.81      0.81      3181

some of the screenshots of graphs and tables from the output

<img width="1086" alt="1" src="https://github.com/nikithams2020/BDA602-Nikitha/assets/102404042/f633b61e-c29e-41b2-bf98-93bcf663e00a">

<img width="1438" alt="2" src="https://github.com/nikithams2020/BDA602-Nikitha/assets/102404042/7edbeafb-24b6-4b57-89a8-170409830620">

<img width="1430" alt="3" src="https://github.com/nikithams2020/BDA602-Nikitha/assets/102404042/906aba00-437e-4ab2-ae57-943e07423635">

<img width="745" alt="4" src="https://github.com/nikithams2020/BDA602-Nikitha/assets/102404042/587765a1-72aa-4a81-a910-fb1de820a370">

<img width="849" alt="5" src="https://github.com/nikithams2020/BDA602-Nikitha/assets/102404042/252e138c-1f04-4fd5-8f18-1c50b703ee5d">

<img width="1440" alt="6" src="https://github.com/nikithams2020/BDA602-Nikitha/assets/102404042/f21dbe4e-704c-4f7e-9ca5-5aa924a834d7">

<img width="1415" alt="7" src="https://github.com/nikithams2020/BDA602-Nikitha/assets/102404042/1cef8b42-bc39-4bf2-932a-fe7aef3b208d">

<img width="1440" alt="8" src="https://github.com/nikithams2020/BDA602-Nikitha/assets/102404042/512a67f9-f030-46e3-9ec1-02e2ac03afae">






