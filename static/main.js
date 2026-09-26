document.addEventListener("DOMContentLoaded", () => {
    const analyzeForm = document.getElementById("analyzeForm");
    const progressSection = document.getElementById("progressSection");
    const submitBtn = document.getElementById("submitBtn");

    if (analyzeForm) {
        analyzeForm.addEventListener("submit", async (e) => {
            e.preventDefault();

            // Form Validation
            const resumeInput = document.getElementById("resume");
            const jobText = document.getElementById("job_description").value.trim();

            if (!resumeInput.files.length || !jobText) {
                alert("Please select a resume file and enter a job description.");
                return;
            }

            // Prepare UI for processing
            submitBtn.disabled = true;
            submitBtn.innerText = "Processing Analysis...";
            progressSection.classList.remove("hidden");

            // Simulate progression ticks visually while request processes
            animateSteps();

            const formData = new FormData();
            formData.append("resume", resumeInput.files[0]);
            formData.append("job_description", jobText);

            try {
                const response = await fetch("/analyze", {
                    method: "POST",
                    body: formData
                });

                const result = await response.json();

                if (result.success && result.redirect_url) {
                    window.location.href = result.redirect_url;
                } else {
                    alert("Analysis Failed: " + (result.error || "Unknown error occurred."));
                    resetUI();
                }
            } catch (err) {
                alert("Server Connection Error: " + err.message);
                resetUI();
            }
        });
    }

    function animateSteps() {
        const steps = ["step-resume", "step-job", "step-matching", "step-gap", "step-final"];
        let currentStep = 0;

        const interval = setInterval(() => {
            if (currentStep < steps.length) {
                const stepEl = document.getElementById(steps[currentStep]);
                if (stepEl) {
                    stepEl.classList.add("active");
                    stepEl.querySelector(".step-icon").innerText = "⚙️";
                }
                if (currentStep > 0) {
                    const prevEl = document.getElementById(steps[currentStep - 1]);
                    if (prevEl) {
                        prevEl.classList.remove("active");
                        prevEl.classList.add("completed");
                        prevEl.querySelector(".step-icon").innerText = "✅";
                    }
                }
                currentStep++;
            } else {
                clearInterval(interval);
            }
        }, 3000);
    }

    function resetUI() {
        submitBtn.disabled = false;
        submitBtn.innerText = "Analyze Match";
        progressSection.classList.add("hidden");
    }
});