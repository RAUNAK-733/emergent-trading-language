const authService = require("../services/authService");
const asyncHandler = require("../utils/asyncHandler");

const register = asyncHandler(async function(req, res) {
  const data = await authService.registerUser(req.body);

  return res.status(201).json({
    success: true,
    message: "User registered successfully",
    data
  });
});

const login = asyncHandler(async function(req, res) {
  const data = await authService.loginUser(req.body);

  return res.json({
    success: true,
    message: "Login successful",
    data
  });
});

const getMe = asyncHandler(async function(req, res) {
  return res.json({
    success: true,
    message: "Current user fetched successfully",
    data: {
      user: req.user
    }
  });
});

module.exports = {
  register,
  login,
  getMe
};
