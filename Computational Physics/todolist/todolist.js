const predefinedTasks = ['Complete all exercises', 'Finish all figures (and make them nice)', 'Make the code readable (including markdown comments', "Setup this week's portfolio"];
const totalWeeks = 14;

// Function to toggle cross-out effect and save the state in localStorage
function toggleCrossOut(checkbox) {
    const text = checkbox.nextElementSibling;
    const taskId = checkbox.id;

    // Toggle cross-out style and save state
    if (checkbox.checked) {
        text.classList.add('crossed-out');
        localStorage.setItem(taskId, 'checked');
    } else {
        text.classList.remove('crossed-out');
        localStorage.setItem(taskId, 'unchecked');
    }

    // Check if the week header should be crossed out
    checkWeekCompletion(checkbox.closest('.task-list'));
}

// Function to dynamically create the task list for 14 weeks
function createTaskList() {
    const taskContainer = document.getElementById('taskContainer');

    // Create tasks for each week
    for (let week = 1; week <= totalWeeks; week++) {
        const weekTitle = document.createElement('div');
        weekTitle.textContent = `Week ${week}`;
        weekTitle.classList.add('week-title');
        weekTitle.id = `week-title-${week}`;
        taskContainer.appendChild(weekTitle);

        const taskList = document.createElement('ul');
        taskList.classList.add('task-list');
        taskList.dataset.week = week; // Add week identifier for tracking

        // Create predefined tasks
        predefinedTasks.forEach((task, index) => {
            createTaskItem(taskList, `week${week}-task${index + 1}`, task);
        });

        taskContainer.appendChild(taskList);
    }

    // Load saved state from localStorage
    loadChecklistState();
}

// Function to create a task item
function createTaskItem(taskList, taskId, taskText) {
    const listItem = document.createElement('li');
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = taskId;
    checkbox.onclick = function () { toggleCrossOut(this); };

    const taskLabel = document.createElement('span');
    taskLabel.textContent = taskText;

    listItem.appendChild(checkbox);
    listItem.appendChild(taskLabel);
    taskList.appendChild(listItem);
}

// Function to load the state of checkboxes from localStorage
function loadChecklistState() {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(checkbox => {
        const taskId = checkbox.id;
        const state = localStorage.getItem(taskId);

        if (state === 'checked') {
            checkbox.checked = true;
            checkbox.nextElementSibling.classList.add('crossed-out');
        } else {
            checkbox.checked = false;
            checkbox.nextElementSibling.classList.remove('crossed-out');
        }
    });

    // Check the completion status of each week after loading states
    document.querySelectorAll('.task-list').forEach(checkWeekCompletion);
}

// Function to check if all tasks in a week are completed and cross out the week header
function checkWeekCompletion(taskList) {
    const weekTitle = document.getElementById(`week-title-${taskList.dataset.week}`);
    const allTasks = taskList.querySelectorAll('input[type="checkbox"]');
    const allChecked = Array.from(allTasks).every(checkbox => checkbox.checked);

    if (allChecked) {
        weekTitle.classList.add('crossed-out');
    } else {
        weekTitle.classList.remove('crossed-out');
    }
}

// Function to add a new task
function addNewTask() {
    const newTaskInput = document.getElementById('newTaskInput');
    const newTaskText = newTaskInput.value.trim();

    if (newTaskText === '') {
        alert('Please enter a task description.');
        return;
    }

    const taskContainer = document.getElementById('taskContainer');

    // Check if there is an "Additional Tasks" section
    let additionalTasksSection = document.getElementById('additionalTasks');
    if (!additionalTasksSection) {
        additionalTasksSection = document.createElement('div');
        additionalTasksSection.id = 'additionalTasks';
        additionalTasksSection.classList.add('week-title');
        additionalTasksSection.textContent = 'Additional Tasks';
        taskContainer.appendChild(additionalTasksSection);

        const additionalTaskList = document.createElement('ul');
        additionalTaskList.classList.add('task-list');
        taskContainer.appendChild(additionalTaskList);
    }

    // Create the new task item
    const taskList = additionalTasksSection.nextElementSibling;
    const taskId = `additional-task-${Date.now()}`; // Unique ID based on timestamp
    createTaskItem(taskList, taskId, newTaskText);

    // Save new task state as unchecked
    localStorage.setItem(taskId, 'unchecked');

    // Clear the input field
    newTaskInput.value = '';
}

// Initialize the task list on page load
document.addEventListener('DOMContentLoaded', createTaskList);