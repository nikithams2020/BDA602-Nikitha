
--  Annual Batting Average
create table ann_bat_avg as (
select
	bc.batter,
	year(ga.local_date) as year,
	round(sum(Hit)/ sum(atBat), 3) as batting_avg
from
	batter_counts bc
left join game ga on
	ga.game_id = bc.game_id
where
	atBat <> 0
group by
	batter,
	year(ga.local_date)
HAVING
	year is not null
order by
	batter);
	
--  Historic Batting Average
create table if not exists hist_bat_avg as (
select
	bc.batter,
	round(sum(Hit)/ sum(atBat), 3) batting_avg
from
	batter_counts bc
left join game ga on
	ga.game_id = bc.game_id
where
	atBat <> 0
group by
	bc.batter
order by
	bc.batter);
	