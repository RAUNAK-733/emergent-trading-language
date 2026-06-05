function errorMiddleware(error, req, res, next) {
  const statusCode = error.statusCode || 500;

  const response = {
    success: false,
    message: error.message || "Internal server error"
  };

  if (process.env.NODE_ENV !== "production") {
    response.stack = error.stack;
  }

  res.status(statusCode).json(response);
}

module.exports = errorMiddleware;
