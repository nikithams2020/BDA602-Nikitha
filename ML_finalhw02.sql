-- Annual Batting Average --
create table if not exists ann_bat_avg as
(select
	bc.batter
	, year(ga.local_date) as year_y
	, round(sum(Hit) / sum(atBat), 3) as batting_avg
	from
		batter_counts bc
		left join game ga on
			ga.game_id = bc.game_id
	where
		atBat != 0
	group by
		batter
		, year(ga.local_date)
	having
		year_y is not null
	order by
		batter)
;

--  Historic Batting Average
create table if not exists hist_bat_avg as (
	select
		bc.batter
		, round(sum(Hit) / sum(atBat), 3) as batting_avg
	from
		batter_counts bc
		left join game ga on
			ga.game_id = bc.game_id
	where
		atBat != 0
	group by
		bc.batter
	order by
		bc.batter)
;

-- Rolling Average of batters
create table if not exists rolling_avg as (
	with temp_1 as (
		select
			bc.batter
			, bc.game_id
			, bc.Hit
			, bc.atBat
			, datediff(g.local_date, (select min(local_date) from game)) as days_diff
		from
			batter_counts bc
			left join game g on
				g.game_id = bc.game_id
		where
			atBat != 0
	)
	, temp_2 as (
		select
			batter
			, game_id
			, days_diff
			, sum(Hit) / sum(atBat) as bat_avg
		from
			temp_1
		group by
			batter
			, game_id
			, days_diff
	)
	select
		t1.batter
		, t1.game_id
        , coalesce((
            select avg(t2.bat_avg)
            from temp_2 as t2
            where t2.days_diff between t1.days_diff - 100 and t1.days_diff - 1
                and t1.batter = t2.batter
        ), 0) as batting_avg
	from
        temp_2 t1)
;
