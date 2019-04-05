use flags;

# update tables
DELIMITER //
CREATE PROCEDURE reload_table_contents
( 
		IN table_name_ VARCHAR(100),
		IN resource_name_ VARCHAR(100)
	)
	BEGIN
		CREATE TABLE temp LIKE table_name_;
        LOAD DATA INFILE resource_name INTO TABLE temp;
        DROP TABLE table_name_;
        ALTER TABLE temp REANME TO table_name_;
	END
DELIMITER ;

CALL reload_table_contents(ccodes, 'C:\\Users\\hornu\\OneDrive\\Documents\\repos\\flags\\data\\clean-data\clean-country-codes.txt')