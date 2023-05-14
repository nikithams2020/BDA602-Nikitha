

create or replace table pitcher_features(
with home as (
select game_id, team_id as home_team_id,
sum(Strikeout) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_rolling_strikeout,
sum(Walk) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_rolling_walk,
sum(Hit) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_rolling_hits,
sum(outsPlayed) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_rolling_outs_played,
sum(pitchesThrown) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_rolling_pitches_thrown,
sum(pitcher) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_pitcher,
sum(atBat) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_atBat,
sum(toBase) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_toBase,
sum(`Single`) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_Single,
sum(`Double`) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_Double,
sum(Groundout) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_Groundout,
sum(DaysSinceLastPitch) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as H_DaysSinceLastPitch
from pitcher_counts pc where homeTeam = 1 and startingPitcher = 1 order by game_id
),
away as (select game_id, team_id as away_team_id,
sum(Strikeout) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_rolling_strikeout,
sum(Walk) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_rolling_walk,
sum(Hit) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_rolling_hits,
sum(outsPlayed) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_rolling_outs_played,
sum(pitchesThrown) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_rolling_pitches_thrown,
sum(pitcher) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_pitcher,
sum(atBat) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_atBat,
sum(toBase) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_toBase,
sum(`Single`) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_Single,
sum(`Double`) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_Double,
sum(Groundout) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_Groundout,
sum(DaysSinceLastPitch) over(partition by team_id order by game_id rows between 3 PRECEDING and 1 PRECEDING) as A_DaysSinceLastPitch
from pitcher_counts pc where awayTeam = 1 and startingPitcher = 1 order by game_id) 
select h.game_id, home_team_id, H_rolling_strikeout, H_rolling_walk, H_rolling_hits,
H_rolling_outs_played, H_rolling_pitches_thrown,H_pitcher,H_atBat, H_toBase, H_Double, H_Groundout, H_DaysSinceLastPitch, away_team_id, A_rolling_strikeout, 
A_rolling_walk, A_rolling_hits, A_rolling_outs_played, A_rolling_pitches_thrown, A_pitcher, A_atBat, A_toBase, A_Double, A_Groundout, A_DaysSinceLastPitch
from home h left join away a on h.game_id = a.game_id);

Create or replace  table base_features  (Select * from pitcher_features pcf inner join 
(select game_id as gameid, away_hits,home_hits,away_errors,home_errors,case when winner_home_or_away='H' then 1 else 0 end  as hometeamwins from boxscore) bc on pcf.game_id = bc.gameid
);
