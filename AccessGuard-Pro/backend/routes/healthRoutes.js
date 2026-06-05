const express = require("express");
const asyncHandler = require("../utils/asyncHandler");

const router = express.Router();

router.get("/", asyncHandler(async function(req, res) {
  res.json({
    success: true,
    message: "AccessGuard Pro API is running",
    data: {
      status: "healthy",
      environment: process.env.NODE_ENV || "development",
      timestamp: new Date().toISOString()
    }
  });
}));

module.exports = router;
