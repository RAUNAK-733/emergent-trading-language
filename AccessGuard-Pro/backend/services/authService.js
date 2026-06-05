const bcrypt = require("bcrypt");
const { pool } = require("../config/db");
const generateToken = require("../utils/generateToken");

const allowedRoles = [
  "Admin",
  "RiskManager",
  "ComplianceOfficer",
  "Auditor",
  "Teacher",
  "Student",
  "Accountant",
  "Guest"
];

function removePasswordHash(user) {
  return {
    id: user.id,
    username: user.username,
    email: user.email,
    role: user.role,
    status: user.status,
    created_at: user.created_at,
    updated_at: user.updated_at
  };
}

async function findUserByEmail(email) {
  const [rows] = await pool.execute(
    "SELECT id, username, email, password_hash, role, status, created_at, updated_at FROM users WHERE email = ? LIMIT 1",
    [email]
  );

  return rows[0] || null;
}

async function getUserById(id) {
  const [rows] = await pool.execute(
    "SELECT id, username, email, role, status, created_at, updated_at FROM users WHERE id = ? LIMIT 1",
    [id]
  );

  return rows[0] || null;
}

async function registerUser(data) {
  const username = data.username.trim().toLowerCase();
  const email = data.email.trim().toLowerCase();
  const password = data.password;
  const role = data.role || "Guest";

  if (!allowedRoles.includes(role)) {
    const error = new Error("Invalid role selected");
    error.statusCode = 400;
    throw error;
  }

  const [existingUsers] = await pool.execute(
    "SELECT id FROM users WHERE username = ? OR email = ? LIMIT 1",
    [username, email]
  );

  if (existingUsers.length > 0) {
    const error = new Error("Username or email already exists");
    error.statusCode = 409;
    throw error;
  }

  const passwordHash = await bcrypt.hash(password, 12);

  const [result] = await pool.execute(
    "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
    [username, email, passwordHash, role]
  );

  const user = await getUserById(result.insertId);
  const token = generateToken(user);

  return {
    user: removePasswordHash(user),
    token
  };
}

async function loginUser(data) {
  const email = data.email.trim().toLowerCase();
  const password = data.password;
  const userWithPassword = await findUserByEmail(email);

  if (!userWithPassword) {
    const error = new Error("Invalid email or password");
    error.statusCode = 401;
    throw error;
  }

  if (userWithPassword.status === "Inactive") {
    const error = new Error("User account is inactive");
    error.statusCode = 403;
    throw error;
  }

  const passwordMatches = await bcrypt.compare(password, userWithPassword.password_hash);

  if (!passwordMatches) {
    const error = new Error("Invalid email or password");
    error.statusCode = 401;
    throw error;
  }

  const user = removePasswordHash(userWithPassword);
  const token = generateToken(user);

  return {
    user,
    token
  };
}

module.exports = {
  allowedRoles,
  registerUser,
  loginUser,
  getUserById
};
