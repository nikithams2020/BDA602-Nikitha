echo "db testing"

if ! mariadb -u root -h mariadb9  -p root -e 'use baseball';
then
  echo "Baseball does not exist"
  mariadb -u root -h mariadb9  -p root -e 'create database if not exists baseball';
  mysql -u root -h mariadb9  -p root < baseball.sql
  mysql -u root -h mariadb9  -p root < baseball_features_sql.sql

else
  echo "Database exists"
  mysql -u root -h mariadb9  -p root < baseball_features_sql.sql

fi

  mysql -u root -h mariadb9  -p root -e 'SELECT * FROM baseball_features'> test.csv
  echo "process completed"


