CREATE OR REPLACE TABLE startingpitcher_rolling_stats (

WITH rolling_stats AS (
  SELECT game_id,
         team_id,
         tm_ref,
         SUM(strikeout) OVER w1 AS roll_strk,
         SUM(walk) OVER w1 AS roll_strk_walk,
         SUM(hit) OVER w1 AS roll_strk_hit,
         SUM(outsplayed) OVER w1 AS roll_strk_outsplayed,
         SUM(pitchesthrown) OVER w1 AS roll_strk_pitchesthrown
  FROM rolling_stat_new
--  
  window w1 as (PARTITION BY team_id ORDER BY game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)
  )  

  SELECT h.game_id,
       h.team_id AS hteam_id,
       h.roll_strk AS h_roll_strikeout,
       h.roll_strk_walk AS h_roll_walk,
       h.roll_strk_hit AS h_roll_hit,
       h.roll_strk_outsplayed AS h_roll_outsplayed,
       h.roll_strk_pitchesthrown AS h_roll_pitchesthrown,
       a.team_id AS away_team_id,
       a.roll_strk AS a_roll_strikeout,
       a.roll_strk_walk AS a_roll_walk,
       a.roll_strk_hit AS a_roll_hit,
       a.roll_strk_outsplayed AS a_roll_outsplayed,
       a.roll_strk_pitchesthrown AS a_roll_pitchesthrown
FROM(

select * from rolling_stats where tm_ref=1
) as h
LEFT JOIN
(
select * from rolling_stats where tm_ref=2
) as a 
 ON   h.game_id = a.game_id
ORDER BY h.game_id
);

CREATE TEMPORARY TABLE rolling_stat_new AS
SELECT *

FROM (
 SELECT game_id,
         team_id,
         strikeout,
         walk,
         hit,
         outsplayed,
         pitchesthrown,
         1 as tm_ref
  FROM pitcher_counts
  WHERE startingpitcher = 1  AND homeTeam = 1
union all
  	SELECT game_id,
          team_id,
         strikeout,
         walk,
         hit, 
         outsplayed,
         pitchesthrown,
         2 as tm_ref
		 FROM pitcher_counts
		 WHERE startingpitcher = 1  AND awayTeam = 1
		 ) AS subquery
		 GROUP BY game_id, team_id;
		
		
SELECT *
FROM startingpitcher_rolling_stats
-- LIMIT 10;

CREATE OR REPLACE TABLE batting_stats_rolling_stats AS (
  SELECT 
    h.game_id, 
    h.home_team_id, 
    h.h_roll_batt_avg, 
    h.h_ttol_bases, 
    a.away_team_id, 
    a.a_roll_batt_avg, 
    a.a_ttol_bases
  FROM (
    SELECT 
      ga.game_id,
      tm_bc.team_id AS home_team_id,
      SUM(tm_bc.hit) OVER w2 / 
      SUM(tm_bc.atBat) OVER w2 AS h_roll_batt_avg,
      SUM(tm_bc.single + 2*tm_bc.double + 3*tm_bc.triple + 4*tm_bc.home_run) OVER w2 AS h_ttol_bases
    FROM game ga
    JOIN team_batting_counts tm_bc ON ga.game_id = tm_bc.game_id AND homeTeam = 1
    WINDOW w2 AS (
      PARTITION BY tm_bc.team_id 
      ORDER BY ga.local_date
    )
  ) h 
  JOIN (
    SELECT 
      ga.game_id,
      tm_bc.team_id AS away_team_id,
      SUM(tm_bc.hit) OVER w2/ 
      SUM(tm_bc.atBat) OVER w2 AS a_roll_batt_avg,
      SUM(tm_bc.single + 2*tm_bc.double + 3*tm_bc.triple + 4*tm_bc.home_run) OVER w2 AS a_ttol_bases
    FROM game ga
LEFT  JOIN team_batting_counts tm_bc ON ga.game_id = tm_bc.game_id AND awayTeam = 1
    WINDOW w2 AS (
      PARTITION BY tm_bc.team_id 
      ORDER BY ga.local_date
    )
  ) a ON h.game_id = a.game_id
  ORDER BY h.game_id
);

  
SELECT *
FROM batting_stats_rolling_stats ;
-- LIMIT 10;

CREATE OR REPLACE TABLE baseball_features AS (
select p.game_id, p.home_team_id,    CASE WHEN bx.winner_home_or_away = 'H' THEN 1 ELSE 0 END AS HomeTeamWins,
b.H_total_bases,b.A_total_bases  ,p.H_rolling_strikeout, p.A_rolling_strikeout, p.H_rolling_hits , p.A_rolling_hits ,bx.home_runs, bx.away_runs, bx.home_hits,
bx.away_hits  from boxscore bx
left join startingPitcher_last_5_games_rolling_stats p on bx.game_id = p.game_id 
left join batting_stats_last_50_days_rolling_stats b on p.game_id = b.game_id order by bx.game_id);


select * from baseball_features
  
  
