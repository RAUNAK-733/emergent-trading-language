const express = require("express");
const { body } = require("express-validator");
const authController = require("../controllers/authController");
const authMiddleware = require("../middleware/authMiddleware");
const validateRequest = require("../middleware/validateMiddleware");
const { authLimiter } = require("../middleware/rateLimitMiddleware");
const { allowedRoles } = require("../services/authService");

const router = express.Router();

router.post(
  "/register",
  authLimiter,
  [
    body("username")
      .trim()
      .notEmpty().withMessage("Username is required")
      .isLength({ min: 3, max: 50 }).withMessage("Username must be 3 to 50 characters"),
    body("email")
      .trim()
      .notEmpty().withMessage("Email is required")
      .isEmail().withMessage("Please provide a valid email"),
    body("password")
      .notEmpty().withMessage("Password is required")
      .isLength({ min: 8 }).withMessage("Password must be at least 8 characters"),
    body("role")
      .optional()
      .isIn(allowedRoles).withMessage("Invalid role selected")
  ],
  validateRequest,
  authController.register
);

router.post(
  "/login",
  authLimiter,
  [
    body("email")
      .trim()
      .notEmpty().withMessage("Email is required")
      .isEmail().withMessage("Please provide a valid email"),
    body("password")
      .notEmpty().withMessage("Password is required")
  ],
  validateRequest,
  authController.login
);

router.get("/me", authMiddleware, authController.me);

module.exports = router;
