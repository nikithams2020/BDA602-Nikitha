#! /bin/bash

echo "db testing"

sleep 150
if ! mariadb -h mariadb101 -u root -proot  -e "use baseball;";
then
  echo "Baseball does not exist"
  mariadb -hmariadb101 -P3306 -uroot -proot -e "create database if not exists baseball;";
  mariadb -hmariadb101 -P3306 -uroot -proot -e 'use baseball;';
  mariadb -hmariadb101 -P3306 -uroot -proot -Dbaseball < ./data/baseball.sql
  mariadb -hmariadb101 -P3306 -uroot -proot -Dbaseball < ./data/baseball_features_sql.sql

else
  echo "Database exists"
  mariadb -hmariadb101 -P3306 -uroot  -proot -e 'use baseball';
  mariadb -hmariadb101 -P3306 -uroot  -proot -Dbaseball < ./data/baseball_features_sql.sql 

fi
  mariadb -hmariadb101 -P3306 -uroot  -proot -Dbaseball -e 'SELECT * FROM base_features' > ./data/test.csv
  echo "Data processing completed"
  python baseball_features.py
  kill -SIGKILL 1 