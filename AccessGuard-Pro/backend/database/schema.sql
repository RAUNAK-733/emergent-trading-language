CREATE DATABASE IF NOT EXISTS accessguard_pro;
USE accessguard_pro;

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) NOT NULL UNIQUE,
  email VARCHAR(100) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role ENUM('Admin','RiskManager','ComplianceOfficer','Auditor','Teacher','Student','Accountant','Guest') NOT NULL DEFAULT 'Guest',
  status ENUM('Active','Inactive') NOT NULL DEFAULT 'Active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_users_role (role),
  INDEX idx_users_status (status)
);

CREATE TABLE IF NOT EXISTS risks (
  id INT AUTO_INCREMENT PRIMARY KEY,
  risk_code VARCHAR(20) NOT NULL UNIQUE,
  title VARCHAR(150) NOT NULL,
  description TEXT,
  impact ENUM('Low','Medium','High') NOT NULL,
  likelihood ENUM('Low','Medium','High') NOT NULL,
  risk_level ENUM('Low','Medium','High') NOT NULL,
  control TEXT,
  owner_id INT NULL,
  status ENUM('Open','In Progress','Mitigated','Closed') DEFAULT 'Open',
  created_by INT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_risks_risk_level (risk_level),
  INDEX idx_risks_status (status),
  CONSTRAINT fk_risks_owner FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE SET NULL,
  CONSTRAINT fk_risks_created_by FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS compliance_controls (
  id INT AUTO_INCREMENT PRIMARY KEY,
  control_code VARCHAR(20) NOT NULL UNIQUE,
  title VARCHAR(150) NOT NULL,
  framework VARCHAR(100),
  status ENUM('Completed','Pending','In Progress') DEFAULT 'Pending',
  purpose TEXT,
  owner_id INT NULL,
  created_by INT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_compliance_status (status),
  CONSTRAINT fk_compliance_owner FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE SET NULL,
  CONSTRAINT fk_compliance_created_by FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  username VARCHAR(50),
  role VARCHAR(50),
  action VARCHAR(150) NOT NULL,
  entity_type VARCHAR(100),
  entity_id INT NULL,
  status ENUM('Allowed','Denied','Success','Failed') NOT NULL,
  ip_address VARCHAR(100),
  user_agent TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_audit_logs_created_at (created_at),
  CONSTRAINT fk_audit_logs_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS incidents (
  id INT AUTO_INCREMENT PRIMARY KEY,
  incident_code VARCHAR(20) NOT NULL UNIQUE,
  title VARCHAR(150) NOT NULL,
  description TEXT,
  severity ENUM('Low','Medium','High','Critical') NOT NULL,
  status ENUM('Open','Investigating','Resolved','Closed') DEFAULT 'Open',
  reported_by INT NULL,
  assigned_to INT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_incidents_status (status),
  INDEX idx_incidents_severity (severity),
  CONSTRAINT fk_incidents_reported_by FOREIGN KEY (reported_by) REFERENCES users(id) ON DELETE SET NULL,
  CONSTRAINT fk_incidents_assigned_to FOREIGN KEY (assigned_to) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS policies (
  id INT AUTO_INCREMENT PRIMARY KEY,
  policy_code VARCHAR(20) NOT NULL UNIQUE,
  title VARCHAR(150) NOT NULL,
  category VARCHAR(100),
  content TEXT,
  status ENUM('Draft','Active','Archived') DEFAULT 'Draft',
  created_by INT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_policies_status (status),
  CONSTRAINT fk_policies_created_by FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);
