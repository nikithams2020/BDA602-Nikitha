CREATE OR REPLACE TABLE startingpitcher_rolling_stats AS
WITH rolling_stats AS (
  SELECT game_id,
         team_id,
         SUM(strikeout) OVER w1 AS roll_strk,
         SUM(walk) OVER w1 AS roll_strk_walk,
         SUM(hit) OVER w1 AS roll_strk_hit,
         SUM(outsplayed) OVER w1 AS roll_strk_outsplayed,
         SUM(pitchesthrown) OVER w1 AS roll_strk_pitchesthrown
  FROM pitcher_counts
  WHERE startingpitcher = 1
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
FROM rolling_stats h
LEFT JOIN rolling_stats a ON h.game_id = a.game_id AND h.team_id <> a.team_id
WHERE h.team_id = homeTeam AND a.team_id = awayTeam
ORDER BY h.game_id;

CREATE TEMPORARY TABLE rolling_stats AS
SELECT game_id,
       team_id,
       SUM(strikeout) AS rolling_strikeout,
       SUM(walk) AS rolling_walk,
       SUM(hit) AS rolling_hit,
       SUM(outsplayed) AS rolling_outsplayed,
       SUM(pitchesthrown) AS rolling_pitchesthrown
FROM (
  SELECT game_id,
         homeTeam AS team_id,
         strikeout,
         walk,
         hit,
         outsplayed,
         pitchesthrown
  FROM pitcher_counts
  WHERE startingpitcher = 1 AND homeTeam = 1
  UNION ALL
  SELECT game_id,
         awayTeam AS team_id,
         strikeout,
         walk,
         hit, 
         outsplayed,
         pitchesthrown
		 FROM pitcher_counts
		 WHERE startingpitcher = 1 AND awayTeam = 1
		 ) AS subquery
		 GROUP BY game_id, team_id;
		
SELECT *
FROM startingpitcher_rolling_stats
LIMIT 10;

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
      ORDER BY DATEDIFF(ga.local_date, MIN(ga.local_date)) 
      RANGE BETWEEN 50 PRECEDING AND 1 PRECEDING
    )
  ) h 
  JOIN (
    SELECT 
      ga.game_id,
      tm_bc.team_id AS away_team_id,
      SUM(tm_bc.hit) OVER w2 / 
      SUM(tm_bc.atBat) OVER w2 AS a_roll_batt_avg,
      SUM(tm_bc.single + 2*tm_bc.double + 3*tm_bc.triple + 4*tm_bc.home_run) OVER w2 AS a_ttol_bases
    FROM game ga
    JOIN team_batting_counts tm_bc ON ga.game_id = tm_bc.game_id AND awayTeam = 1
    WINDOW w2 AS (
      PARTITION BY tm_bc.team_id 
      ORDER BY DATEDIFF(ga.local_date, MIN(ga.local_date)) 
      RANGE BETWEEN 50 PRECEDING AND 1 PRECEDING
    )
  ) a ON h.game_id = a.game_id
  ORDER BY h.game_id
);
  
SELECT *
FROM batting_stats_rolling_stats 
LIMIT 10;

CREATE OR REPLACE TABLE baseball_features AS (
  SELECT 
    score.game_id,
    CASE WHEN score.winner_home_or_away = 'H' THEN 1 ELSE 0 END AS HomeTeamWins,
    sp.home_team_id,
    sp.rolling_strikeout AS H_rolling_strikeout,
    sp.rolling_walk AS H_rolling_walk,
    sp.rolling_hits AS H_rolling_hits,
    sp.rolling_outs_played AS H_rolling_outs_played,
    sp.rolling_pitches_thrown AS H_rolling_pitches_thrown,
    sp.away_team_id,
    sp_away.rolling_strikeout AS A_rolling_strikeout,
    sp_away.rolling_walk AS A_rolling_walk,
    sp_away.rolling_hits AS A_rolling_hits,
    sp_away.rolling_outs_played AS A_rolling_outs_played,
    sp_away.rolling_pitches_thrown AS A_rolling_pitches_thrown,
    bs_home.rolling_batting_avg AS H_rolling_batting_avg,
    bs_home.total_bases AS H_total_bases,
    bs_away.rolling_batting_avg AS A_rolling_batting_avg,
    bs_away.total_bases AS A_total_bases
  FROM 
    score_table AS score
    JOIN startingpitcher_rolling_stats AS sp ON score.game_id = sp.game_id
    JOIN startingpitcher_rolling_stats AS sp_away ON score.game_id = sp_away.game_id AND sp.home_team_id <> sp_away.home_team_id
    JOIN batting_stats_rolling_stats AS bs_home ON sp.home_team_id = bs_home.team_id AND score.date >= bs_home.date_start AND score.date <= bs_home.date_end
    JOIN batting_stats_rolling_stats AS bs_away ON sp_away.away_team_id = bs_away.team_id AND score.date >= bs_away.date_start AND score.date <= bs_away.date_end
  ORDER BY score.game_id
);

