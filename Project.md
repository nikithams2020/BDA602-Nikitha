Data Problems
In the two tables there were many columns with same name it created some confusions
Eg: Fly_Out and flyout 
Force_Out and force_out
Tried to leave some the columns as it could cheat 
We also see that that home_streak and away_streak columns tend to cheat the prediction.

Feature Selection
Tables – Pitcher Counts  and BoxScore
Response- Home Team Wins
WHY these Tables?
Pitcher Counts
A pitcher's main objective is to prevent the opposing team from scoring runs, and they do this by throwing pitches that are difficult for the batter to hit or by inducing ground balls or fly balls that can be easily fielded by the defensive team.
Box Score
Game Statistics and also player’s performances and Team’s performances

Analyzing Baseball Dataset
Initial Features and Tables
Tables – Pitcher Counts  and BoxScore
Response- Home Team Wins
Features w.r.t Home team:
Strikeout
Walk
OutsPlayed
Pitches Thrown
Pitcher
atBat
toBase
Single
Double
Groundout
DaysSinceLastPitch
Features w.r.t Away team:
Strikeout
Walk
OutsPlayed
Pitches Thrown
Pitcher
atBat
toBase
Single
Double
Groundout
DaysSinceLastPitch
Features w.r.t Boxscore:
home_runs
home_hits
away_runs
away_hits
away_errors
home_errors
Own Features:
home_runrate [home_runs/home_hits] 
Away_runrate [ away_runs/away_hits]
Home_Error_Rate/Home_hits
Away_Error_Rate/Away_hits



