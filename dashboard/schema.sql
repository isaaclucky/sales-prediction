CREATE TABLE IF NOT EXISTS `SalesInfo` (
    `id` INT NOT NULL AUTO_INCREMENT,
    `Store` INT DEFAULT NULL,
    `DayOfWeek` INT DEFAULT NULL,
    `Sales` INT DEFAULT NULL,
    `Open` INT DEFAULT NULL,
    `Promo` INT DEFAULT NULL,
    `StateHoliday` CHAR DEFAULT NULL,
    `SchoolHoliday` INT DEFAULT NULL,
    `StoreType` TEXT DEFAULT NULL,
    `Assortment` TEXT DEFAULT NULL,
    `CompetitionDistance` FLOAT DEFAULT NULL,
    `Promo2` INT DEFAULT NULL,
    `PromoInterval` TEXT DEFAULT NULL,
    `Until_Holiday` INT DEFAULT NULL,
    `Since_Holiday` INT DEFAULT NULL,
    `Year` INT DEFAULT NULL,
    `Month` INT DEFAULT NULL,
    `Quarter` INT DEFAULT NULL,
    `Week` INT DEFAULT NULL,
    `Day` INT DEFAULT NULL,
    `WeekOfYear` INT DEFAULT NULL,
    `DayOfYear` INT DEFAULT NULL,
    `IsWeekDay` INT DEFAULT NULL,
    `CompetitionOpenMonthDuration` FLOAT DEFAULT NULL,
    `PromoOpenMonthDuration` FLOAT DEFAULT NULL,
    `Season` TEXT DEFAULT NULL,
    `Month_Status` TEXT DEFAULT NULL,





    PRIMARY KEY (`id`)
) ;