const authService = require("../services/authService");

async function register(req, res, next) {
  try {
    const data = await authService.registerUser(req.body);

    return res.status(201).json({
      success: true,
      message: "User registered successfully",
      data
    });
  } catch (error) {
    next(error);
  }
}

async function login(req, res, next) {
  try {
    const data = await authService.loginUser(req.body);

    return res.json({
      success: true,
      message: "Login successful",
      data
    });
  } catch (error) {
    next(error);
  }
}

async function me(req, res) {
  return res.json({
    success: true,
    message: "Current user fetched successfully",
    data: {
      user: req.user
    }
  });
}

module.exports = {
  register,
  login,
  me
};
