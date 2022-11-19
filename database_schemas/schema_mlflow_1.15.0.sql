-- MySQL dump 10.13  Distrib 5.7.31, for macos10.14 (x86_64)
--
-- Host: 127.0.0.1    Database: mlflow
-- ------------------------------------------------------
-- Server version	5.7.31


--
-- Table structure for table `alembic_version`
--

DROP TABLE IF EXISTS `alembic_version`;
CREATE TABLE `alembic_version` (
  `version_num` varchar(32) NOT NULL,
  PRIMARY KEY (`version_num`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `experiment_tags`
--

DROP TABLE IF EXISTS `experiment_tags`;
CREATE TABLE `experiment_tags` (
  `key` varchar(250) NOT NULL,
  `value` varchar(5000) DEFAULT NULL,
  `experiment_id` int(11) NOT NULL,
  PRIMARY KEY (`key`,`experiment_id`),
  KEY `experiment_id` (`experiment_id`),
  CONSTRAINT `experiment_tags_ibfk_1` FOREIGN KEY (`experiment_id`) REFERENCES `experiments` (`experiment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `experiments`
--

DROP TABLE IF EXISTS `experiments`;
CREATE TABLE `experiments` (
  `experiment_id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(256) NOT NULL,
  `artifact_location` varchar(256) DEFAULT NULL,
  `lifecycle_stage` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`experiment_id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `latest_metrics`
--

DROP TABLE IF EXISTS `latest_metrics`;
CREATE TABLE `latest_metrics` (
  `key` varchar(250) NOT NULL,
  `value` double NOT NULL,
  `timestamp` bigint(20) DEFAULT NULL,
  `step` bigint(20) NOT NULL,
  `is_nan` tinyint(1) NOT NULL,
  `run_uuid` varchar(32) NOT NULL,
  PRIMARY KEY (`key`,`run_uuid`),
  KEY `run_uuid` (`run_uuid`),
  CONSTRAINT `latest_metrics_ibfk_1` FOREIGN KEY (`run_uuid`) REFERENCES `runs` (`run_uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `metrics`
--

DROP TABLE IF EXISTS `metrics`;
CREATE TABLE `metrics` (
  `key` varchar(250) NOT NULL,
  `value` double NOT NULL,
  `timestamp` bigint(20) NOT NULL,
  `run_uuid` varchar(32) NOT NULL,
  `step` bigint(20) NOT NULL DEFAULT '0',
  `is_nan` tinyint(1) NOT NULL,
  PRIMARY KEY (`key`,`timestamp`,`step`,`run_uuid`,`value`,`is_nan`),
  KEY `run_uuid` (`run_uuid`),
  CONSTRAINT `metrics_ibfk_1` FOREIGN KEY (`run_uuid`) REFERENCES `runs` (`run_uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `model_version_tags`
--

DROP TABLE IF EXISTS `model_version_tags`;
CREATE TABLE `model_version_tags` (
  `key` varchar(250) NOT NULL,
  `value` varchar(5000) DEFAULT NULL,
  `name` varchar(256) NOT NULL,
  `version` int(11) NOT NULL,
  PRIMARY KEY (`key`,`name`,`version`),
  KEY `name` (`name`,`version`),
  CONSTRAINT `model_version_tags_ibfk_1` FOREIGN KEY (`name`, `version`) REFERENCES `model_versions` (`name`, `version`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `model_versions`
--

DROP TABLE IF EXISTS `model_versions`;
CREATE TABLE `model_versions` (
  `name` varchar(256) NOT NULL,
  `version` int(11) NOT NULL,
  `creation_time` bigint(20) DEFAULT NULL,
  `last_updated_time` bigint(20) DEFAULT NULL,
  `description` varchar(5000) DEFAULT NULL,
  `user_id` varchar(256) DEFAULT NULL,
  `current_stage` varchar(20) DEFAULT NULL,
  `source` varchar(500) DEFAULT NULL,
  `run_id` varchar(32) DEFAULT NULL,
  `status` varchar(20) DEFAULT NULL,
  `status_message` varchar(500) DEFAULT NULL,
  `run_link` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`name`,`version`),
  CONSTRAINT `model_versions_ibfk_1` FOREIGN KEY (`name`) REFERENCES `registered_models` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `params`
--

DROP TABLE IF EXISTS `params`;
CREATE TABLE `params` (
  `key` varchar(250) NOT NULL,
  `value` varchar(250) NOT NULL,
  `run_uuid` varchar(32) NOT NULL,
  PRIMARY KEY (`key`,`run_uuid`),
  KEY `run_uuid` (`run_uuid`),
  CONSTRAINT `params_ibfk_1` FOREIGN KEY (`run_uuid`) REFERENCES `runs` (`run_uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `registered_model_tags`
--

DROP TABLE IF EXISTS `registered_model_tags`;
CREATE TABLE `registered_model_tags` (
  `key` varchar(250) NOT NULL,
  `value` varchar(5000) DEFAULT NULL,
  `name` varchar(256) NOT NULL,
  PRIMARY KEY (`key`,`name`),
  KEY `name` (`name`),
  CONSTRAINT `registered_model_tags_ibfk_1` FOREIGN KEY (`name`) REFERENCES `registered_models` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `registered_models`
--

DROP TABLE IF EXISTS `registered_models`;
CREATE TABLE `registered_models` (
  `name` varchar(256) NOT NULL,
  `creation_time` bigint(20) DEFAULT NULL,
  `last_updated_time` bigint(20) DEFAULT NULL,
  `description` varchar(5000) DEFAULT NULL,
  PRIMARY KEY (`name`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `runs`
--

DROP TABLE IF EXISTS `runs`;
CREATE TABLE `runs` (
  `run_uuid` varchar(32) NOT NULL,
  `name` varchar(250) DEFAULT NULL,
  `source_type` varchar(20) DEFAULT NULL,
  `source_name` varchar(500) DEFAULT NULL,
  `entry_point_name` varchar(50) DEFAULT NULL,
  `user_id` varchar(256) DEFAULT NULL,
  `status` varchar(9) DEFAULT NULL,
  `start_time` bigint(20) DEFAULT NULL,
  `end_time` bigint(20) DEFAULT NULL,
  `source_version` varchar(50) DEFAULT NULL,
  `lifecycle_stage` varchar(20) DEFAULT NULL,
  `artifact_uri` varchar(200) DEFAULT NULL,
  `experiment_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`run_uuid`),
  KEY `experiment_id` (`experiment_id`),
  CONSTRAINT `runs_ibfk_1` FOREIGN KEY (`experiment_id`) REFERENCES `experiments` (`experiment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Table structure for table `tags`
--

DROP TABLE IF EXISTS `tags`;
CREATE TABLE `tags` (
  `key` varchar(250) NOT NULL,
  `value` varchar(5000) DEFAULT NULL,
  `run_uuid` varchar(32) NOT NULL,
  PRIMARY KEY (`key`,`run_uuid`),
  KEY `run_uuid` (`run_uuid`),
  CONSTRAINT `tags_ibfk_1` FOREIGN KEY (`run_uuid`) REFERENCES `runs` (`run_uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


-- Dump completed on 2021-04-10 14:15:06
