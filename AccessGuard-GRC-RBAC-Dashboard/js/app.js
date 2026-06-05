const rolePermissions = {
  Admin: ["all"],
  Teacher: ["edit_marks", "view_students", "view_attendance", "view_notices"],
  Student: ["view_profile", "view_marks", "view_notices"],
  Accountant: ["view_fees", "update_fees", "view_notices"],
  Guest: ["view_notices"]
};

const featureNames = {
  manage_users: "Manage Users",
  edit_marks: "Edit Marks",
  view_fees: "View Fees",
  view_notices: "View Notices",
  view_attendance: "View Attendance",
  view_students: "View Students",
  update_fees: "Update Fees",
  view_profile: "View Profile",
  view_marks: "View Marks"
};

function getRegisteredUsers() {
  return JSON.parse(localStorage.getItem("users")) || [];
}

function saveRegisteredUsers(users) {
  localStorage.setItem("users", JSON.stringify(users));
}

function registerUser(event) {
  event.preventDefault();

  const usernameInput = document.getElementById("registerUsername");
  const passwordInput = document.getElementById("registerPassword");
  const confirmPasswordInput = document.getElementById("confirmPassword");
  const roleInput = document.getElementById("registerRole");
  const registerMessage = document.getElementById("registerMessage");

  const username = usernameInput.value.trim().toLowerCase();
  const password = passwordInput.value;
  const confirmPassword = confirmPasswordInput.value;
  const role = roleInput.value;

  if (username === "" || password === "" || confirmPassword === "" || role === "") {
    registerMessage.textContent = "Please fill in all fields.";
    return;
  }

  if (password !== confirmPassword) {
    registerMessage.textContent = "Passwords do not match.";
    return;
  }

  const users = getRegisteredUsers();
  const usernameExists = users.some(function(user) {
    return user.username === username;
  });

  if (usernameExists) {
    registerMessage.textContent = "Username is already registered.";
    return;
  }

  const newUser = {
    username: username,
    password: password,
    role: role
  };

  users.push(newUser);
  saveRegisteredUsers(users);

  alert("Registration successful. Please login.");
  window.location.href = "index.html";
}

function loginUser(event) {
  event.preventDefault();

  const usernameInput = document.getElementById("username");
  const passwordInput = document.getElementById("password");
  const loginMessage = document.getElementById("loginMessage");

  const username = usernameInput.value.trim().toLowerCase();
  const password = passwordInput.value;

  if (username === "" || password === "") {
    loginMessage.textContent = "Please enter your username and password.";
    return;
  }

  const users = getRegisteredUsers();
  const matchedUser = users.find(function(user) {
    return user.username === username && user.password === password;
  });

  if (!matchedUser) {
    loginMessage.textContent = "";
    alert("Invalid username or password");
    return;
  }

  localStorage.setItem("accessGuardUsername", matchedUser.username);
  localStorage.setItem("accessGuardRole", matchedUser.role);

  window.location.href = "dashboard.html";
}

function logoutUser() {
  localStorage.removeItem("accessGuardUsername");
  localStorage.removeItem("accessGuardRole");
  window.location.href = "index.html";
}

function getUserData() {
  const username = localStorage.getItem("accessGuardUsername");
  const role = localStorage.getItem("accessGuardRole");

  return {
    username: username,
    role: role
  };
}

function displayUserInfo() {
  const user = getUserData();

  if (!user.username || !user.role) {
    window.location.href = "index.html";
    return;
  }

  const usernameElement = document.getElementById("displayUsername");
  const roleElement = document.getElementById("displayRole");

  if (usernameElement) {
    usernameElement.textContent = user.username;
  }

  if (roleElement) {
    roleElement.textContent = user.role;
  }
}

function checkPermission(feature) {
  const user = getUserData();

  if (!user.username || !user.role) {
    window.location.href = "index.html";
    return;
  }

  const permissions = rolePermissions[user.role] || [];
  const action = featureNames[feature] || feature;

  if (permissions.includes("all") || permissions.includes(feature)) {
    saveAuditLog(user.username, user.role, action, "Allowed");
    alert("Access Allowed");
  } else {
    saveAuditLog(user.username, user.role, action, "Denied");
    alert("Access Denied: You do not have permission.");
  }
}

function saveAuditLog(username, role, action, status) {
  const existingLogs = JSON.parse(localStorage.getItem("accessGuardAuditLogs")) || [];

  const newLog = {
    id: existingLogs.length + 1,
    username: username,
    role: role,
    action: action,
    status: status,
    time: new Date().toLocaleString()
  };

  existingLogs.push(newLog);
  localStorage.setItem("accessGuardAuditLogs", JSON.stringify(existingLogs));
}

function displayAuditLogs() {
  const tableBody = document.getElementById("auditLogTableBody");

  if (!tableBody) {
    return;
  }

  const logs = JSON.parse(localStorage.getItem("accessGuardAuditLogs")) || [];

  if (logs.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="6" class="text-center text-muted">No audit logs found</td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = "";

  logs.forEach(function(log) {
    const statusBadge = log.status === "Allowed"
      ? '<span class="badge bg-success">Allowed</span>'
      : '<span class="badge bg-danger">Denied</span>';

    tableBody.innerHTML += `
      <tr>
        <td>${log.id}</td>
        <td>${log.username}</td>
        <td>${log.role}</td>
        <td>${log.action}</td>
        <td>${statusBadge}</td>
        <td>${log.time}</td>
      </tr>
    `;
  });
}

function getRisks() {
  return JSON.parse(localStorage.getItem("risks")) || [];
}

function saveRisks(risks) {
  localStorage.setItem("risks", JSON.stringify(risks));
}

function addRisk(event) {
  event.preventDefault();

  const user = getUserData();

  if (!user.username || !user.role) {
    window.location.href = "index.html";
    return;
  }

  if (user.role !== "Admin") {
    saveAuditLog(user.username, user.role, "Tried to add risk", "Denied");
    alert("Access Denied: Only Admin can manage risks.");
    return;
  }

  const riskName = document.getElementById("riskName").value.trim();
  const impact = document.getElementById("riskImpact").value;
  const likelihood = document.getElementById("riskLikelihood").value;
  const riskLevel = document.getElementById("riskLevel").value;
  const control = document.getElementById("riskControl").value.trim();

  if (riskName === "" || impact === "" || likelihood === "" || riskLevel === "" || control === "") {
    alert("Please fill in all risk fields.");
    return;
  }

  const risks = getRisks();
  const riskId = "R" + String(risks.length + 1).padStart(3, "0");

  const newRisk = {
    id: riskId,
    name: riskName,
    impact: impact,
    likelihood: likelihood,
    level: riskLevel,
    control: control
  };

  risks.push(newRisk);
  saveRisks(risks);
  saveAuditLog(user.username, user.role, "Added risk", "Allowed");

  document.getElementById("riskName").value = "";
  document.getElementById("riskImpact").value = "";
  document.getElementById("riskLikelihood").value = "";
  document.getElementById("riskLevel").value = "";
  document.getElementById("riskControl").value = "";

  displayRisks();
  alert("Risk added successfully.");
}

function displayRisks() {
  const tableBody = document.getElementById("riskTableBody");

  if (!tableBody) {
    return;
  }

  const risks = getRisks();

  if (risks.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="7" class="text-center text-muted">No risks found</td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = "";

  risks.forEach(function(risk, index) {
    let badgeClass = "bg-success";

    if (risk.level === "High") {
      badgeClass = "bg-danger";
    } else if (risk.level === "Medium") {
      badgeClass = "bg-warning text-dark";
    }

    tableBody.innerHTML += `
      <tr>
        <td>${risk.id}</td>
        <td>${risk.name}</td>
        <td>${risk.impact}</td>
        <td>${risk.likelihood}</td>
        <td><span class="badge ${badgeClass}">${risk.level}</span></td>
        <td>${risk.control}</td>
        <td>
          <button class="btn btn-sm btn-danger" onclick="deleteRisk(${index})">Delete</button>
        </td>
      </tr>
    `;
  });
}

function deleteRisk(index) {
  const user = getUserData();

  if (!user.username || !user.role) {
    window.location.href = "index.html";
    return;
  }

  if (user.role !== "Admin") {
    saveAuditLog(user.username, user.role, "Tried to delete risk", "Denied");
    alert("Access Denied: Only Admin can manage risks.");
    return;
  }

  const risks = getRisks();
  risks.splice(index, 1);
  saveRisks(risks);
  saveAuditLog(user.username, user.role, "Deleted risk", "Allowed");
  displayRisks();
}

function getComplianceControls() {
  return JSON.parse(localStorage.getItem("complianceControls")) || [];
}

function saveComplianceControls(controls) {
  localStorage.setItem("complianceControls", JSON.stringify(controls));
}

function addComplianceControl(event) {
  event.preventDefault();

  const user = getUserData();

  if (!user.username || !user.role) {
    window.location.href = "index.html";
    return;
  }

  if (user.role !== "Admin") {
    saveAuditLog(user.username, user.role, "Tried to manage compliance control", "Denied");
    alert("Access Denied: Only Admin can manage compliance controls.");
    return;
  }

  const controlName = document.getElementById("controlName").value.trim();
  const controlStatus = document.getElementById("controlStatus").value;
  const controlPurpose = document.getElementById("controlPurpose").value.trim();

  if (controlName === "" || controlStatus === "" || controlPurpose === "") {
    alert("Please fill in all compliance control fields.");
    return;
  }

  const controls = getComplianceControls();
  const controlId = "C" + String(controls.length + 1).padStart(3, "0");

  const newControl = {
    id: controlId,
    name: controlName,
    status: controlStatus,
    purpose: controlPurpose
  };

  controls.push(newControl);
  saveComplianceControls(controls);
  saveAuditLog(user.username, user.role, "Added compliance control", "Allowed");

  document.getElementById("controlName").value = "";
  document.getElementById("controlStatus").value = "";
  document.getElementById("controlPurpose").value = "";

  displayComplianceControls();
  alert("Compliance control added successfully.");
}

function displayComplianceControls() {
  const tableBody = document.getElementById("complianceTableBody");

  if (!tableBody) {
    return;
  }

  const controls = getComplianceControls();

  if (controls.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="5" class="text-center text-muted">No compliance controls found</td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = "";

  controls.forEach(function(control, index) {
    let badgeClass = "bg-warning text-dark";

    if (control.status === "Completed") {
      badgeClass = "bg-success";
    } else if (control.status === "In Progress") {
      badgeClass = "bg-info text-dark";
    }

    tableBody.innerHTML += `
      <tr>
        <td>${control.id}</td>
        <td>${control.name}</td>
        <td><span class="badge ${badgeClass}">${control.status}</span></td>
        <td>${control.purpose}</td>
        <td>
          <select class="form-select form-select-sm mb-2" onchange="updateComplianceStatus(${index}, this.value)">
            <option value="Completed" ${control.status === "Completed" ? "selected" : ""}>Completed</option>
            <option value="Pending" ${control.status === "Pending" ? "selected" : ""}>Pending</option>
            <option value="In Progress" ${control.status === "In Progress" ? "selected" : ""}>In Progress</option>
          </select>
          <button class="btn btn-sm btn-danger" onclick="deleteComplianceControl(${index})">Delete</button>
        </td>
      </tr>
    `;
  });
}

function deleteComplianceControl(index) {
  const user = getUserData();

  if (!user.username || !user.role) {
    window.location.href = "index.html";
    return;
  }

  if (user.role !== "Admin") {
    saveAuditLog(user.username, user.role, "Tried to manage compliance control", "Denied");
    alert("Access Denied: Only Admin can manage compliance controls.");
    return;
  }

  const controls = getComplianceControls();
  controls.splice(index, 1);
  saveComplianceControls(controls);
  saveAuditLog(user.username, user.role, "Deleted compliance control", "Allowed");
  displayComplianceControls();
}

function updateComplianceStatus(index, newStatus) {
  const user = getUserData();

  if (!user.username || !user.role) {
    window.location.href = "index.html";
    return;
  }

  if (user.role !== "Admin") {
    saveAuditLog(user.username, user.role, "Tried to manage compliance control", "Denied");
    alert("Access Denied: Only Admin can manage compliance controls.");
    displayComplianceControls();
    return;
  }

  const controls = getComplianceControls();

  if (!controls[index]) {
    return;
  }

  controls[index].status = newStatus;
  saveComplianceControls(controls);
  saveAuditLog(user.username, user.role, "Updated compliance status", "Allowed");
  displayComplianceControls();
}

function checkAdminAccessForUsersPage() {
  const user = getUserData();

  if (!user.username || !user.role) {
    window.location.href = "index.html";
    return;
  }

  const deniedSection = document.getElementById("usersAccessDenied");
  const tableSection = document.getElementById("usersTableSection");

  if (user.role !== "Admin") {
    if (deniedSection) {
      deniedSection.classList.remove("d-none");
    }

    if (tableSection) {
      tableSection.classList.add("d-none");
    }

    saveAuditLog(user.username, user.role, "Tried to access user management", "Denied");
    return;
  }

  if (deniedSection) {
    deniedSection.classList.add("d-none");
  }

  if (tableSection) {
    tableSection.classList.remove("d-none");
  }

  displayUsers();
}

function displayUsers() {
  const tableBody = document.getElementById("usersTableBody");

  if (!tableBody) {
    return;
  }

  const users = getRegisteredUsers();

  if (users.length === 0) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="5" class="text-center text-muted">No users found</td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = "";

  users.forEach(function(user, index) {
    tableBody.innerHTML += `
      <tr>
        <td>U${String(index + 1).padStart(3, "0")}</td>
        <td>${user.username}</td>
        <td><span class="role-badge">${user.role}</span></td>
        <td>
          <select class="form-select form-select-sm" onchange="updateUserRole(${index}, this.value)">
            <option value="Admin" ${user.role === "Admin" ? "selected" : ""}>Admin</option>
            <option value="Teacher" ${user.role === "Teacher" ? "selected" : ""}>Teacher</option>
            <option value="Student" ${user.role === "Student" ? "selected" : ""}>Student</option>
            <option value="Accountant" ${user.role === "Accountant" ? "selected" : ""}>Accountant</option>
            <option value="Guest" ${user.role === "Guest" ? "selected" : ""}>Guest</option>
          </select>
        </td>
        <td>
          <button class="btn btn-sm btn-danger" onclick="deleteUser(${index})">Delete</button>
        </td>
      </tr>
    `;
  });
}

function updateUserRole(index, newRole) {
  const currentUser = getUserData();

  if (!currentUser.username || currentUser.role !== "Admin") {
    saveAuditLog(currentUser.username || "Unknown", currentUser.role || "Unknown", "Tried to manage users", "Denied");
    alert("Access Denied: Only Admin can manage users.");
    return;
  }

  const users = getRegisteredUsers();

  if (!users[index]) {
    return;
  }

  users[index].role = newRole;
  saveRegisteredUsers(users);
  saveAuditLog(currentUser.username, currentUser.role, "Updated user role", "Allowed");

  if (users[index].username === currentUser.username) {
    localStorage.setItem("accessGuardRole", newRole);
    displayUserInfo();
  }

  displayUsers();
}

function deleteUser(index) {
  const currentUser = getUserData();

  if (!currentUser.username || currentUser.role !== "Admin") {
    saveAuditLog(currentUser.username || "Unknown", currentUser.role || "Unknown", "Tried to manage users", "Denied");
    alert("Access Denied: Only Admin can manage users.");
    return;
  }

  const users = getRegisteredUsers();
  const userToDelete = users[index];

  if (!userToDelete) {
    return;
  }

  if (userToDelete.username === currentUser.username) {
    const confirmed = confirm("You are about to delete your own account. Are you sure?");

    if (!confirmed) {
      displayUsers();
      return;
    }
  }

  users.splice(index, 1);
  saveRegisteredUsers(users);
  saveAuditLog(currentUser.username, currentUser.role, "Deleted user", "Allowed");

  if (userToDelete.username === currentUser.username) {
    logoutUser();
    return;
  }

  displayUsers();
}
