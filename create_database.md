# Korean Lottery Database Project Setup Guide

## Project Overview
This project crawls Korean lottery (Lotto 6/45) data and stores it in a MariaDB database hosted on AWS Lightsail.

## Database Setup

### 1. AWS Lightsail Configuration
```bash
# Install Bitnami LAMP stack on AWS Lightsail
# Access MariaDB
mysql -u root -p
```

### 2. Create Database and User
```sql
CREATE DATABASE lotto;
CREATE USER 'admin'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON lotto.* TO 'admin'@'%';
FLUSH PRIVILEGES;
```

### 3. Create Table
```sql
USE lotto;
CREATE TABLE `lotto` (
  `count` int unsigned NOT NULL,
  `1` int NOT NULL,
  `2` int NOT NULL,
  `3` int NOT NULL,
  `4` int NOT NULL,
  `5` int NOT NULL,
  `6` int NOT NULL,
  `7` int NOT NULL,
  `person` int NOT NULL,
  `amount` varchar(45) NOT NULL,
  PRIMARY KEY (`count`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## Python Environment Setup

### 1. Install Required Packages
```bash
pip install requests beautifulsoup4 pymysql pandas matplotlib python-dotenv lxml
```

### 2. Configure Environment Variables
Create `.env` file:
```
DB_HOST=your_host
DB_PORT=3306
DB_USER=admin
DB_PASSWORD=your_password
DB_NAME=lotto
```

## Data Crawling and Storage

### 1. Run Crawler
```bash
python lotto_crawling.py
```

The script will:
- Fetch latest lottery results
- Compare with database
- Insert new data if available
- Generate statistical visualizations

### 2. Verify Data
```sql
-- Check latest records
SELECT * FROM lotto ORDER BY count DESC LIMIT 5;

-- Count total records
SELECT COUNT(*) FROM lotto;
```

## Security Considerations
- Secure MariaDB port (3306)
- Configure AWS security groups
- Set appropriate file permissions for `.env`
- Regular database backups

## Future Enhancements
- API development for data access
- WordPress integration
- Advanced statistical analysis
- Automated data updates

## Author
Jeong Taek Bang

## Project Timeline
January 2024 - Present