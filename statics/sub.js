// Window load handler
window.onload = () => {
    // Removed initThreeJS();
    // Removed animate();
    // Removed setupEventListeners() as its primary function was related to Three.js model interaction
    // The event listeners for the hamburger menu and form are handled by DOMContentLoaded below.
};

// Removed initThreeJS function
// Removed loadFallbackModel function
// Removed animate function
// Removed onWindowResize function

// Applies system theme (dark/light mode)
const applySystemTheme = () => {
    window.matchMedia('(prefers-color-scheme: dark)').matches
        ? document.documentElement.classList.add('dark')
        : document.documentElement.classList.remove('dark');
};
applySystemTheme(); // Apply on initial load
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applySystemTheme);

// Sets up the back-to-top button functionality
function setupBackToTop() {
    const backToTopBtn = document.getElementById('backToTop');
    if (!backToTopBtn) {
        console.error("Back to Top button not found!");
        return;
    }

    const toggleButtonVisibility = () => {
        window.pageYOffset > 300
            ? (backToTopBtn.classList.remove('opacity-0', 'invisible'), backToTopBtn.classList.add('opacity-100', 'visible'))
            : (backToTopBtn.classList.add('opacity-0', 'invisible'), backToTopBtn.classList.remove('opacity-100', 'visible'));
    };

    window.addEventListener('scroll', toggleButtonVisibility);

    backToTopBtn.addEventListener('click', (e) => {
        e.preventDefault();
        window.scrollTo({ top: 0, behavior: 'smooth' }); // Modern way to scroll
        backToTopBtn.blur(); // Remove focus
    });

    backToTopBtn.addEventListener('keydown', (e) => {
        (e.key === "Enter" || e.key === " ") && (
            e.preventDefault(),
            window.scrollTo({ top: 0, behavior: 'smooth' })
        );
    });

    // Initial check for visibility on load
    toggleButtonVisibility();
}

// Placeholder for setupForm function (if it exists elsewhere)
function setupForm() {
    // Your form setup logic here, if any
}

// DOMContentLoaded handler for general setups
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('pageForm');
    const formStatus = document.getElementById('formStatus');

    if (form) { // Ensure the form element exists before adding listener
        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch(form.action, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                // Determine class and message based on response
                const isSuccess = response.ok;
                const statusClass = isSuccess ? 'success' : 'error';
                const initialMessage = 'Request has been send...';

                // Remove conflicting classes and add the correct one
                formStatus.classList.remove(isSuccess ? 'error' : 'success');
                formStatus.classList.add(statusClass);
                formStatus.style.display = 'block'; // Ensure message is displayed

                if (isSuccess) {
                    formStatus.textContent = initialMessage;
                    // Redirect part remains here as it's a side effect
                    setTimeout(() => {
                        window.location.href = './success.html';
                    }, 2000);
                } else {
                    const errorData = await response.json();
                    formStatus.textContent = errorData.message || 'There was an error processing your request. Please try again later.';
                }

            } catch (networkError) {
                console.error('Network or CORS error:', networkError);
                formStatus.classList.remove('success');
                formStatus.classList.add('error');
                formStatus.textContent = 'There was a network error. Please check your connection and try again.';
                formStatus.style.display = 'block';
            }
        });
    }

    setupBackToTop();

    // Permission banner logic
    const permissionBanner = document.getElementById('permissionBanner');
    const acceptPermissionBanner = document.getElementById('acceptPermissionBanner');
    const declinePermissionBanner = document.getElementById('declinePermissionBanner');
    const cacheReminderBanner = document.getElementById('cacheReminderBanner');
    const dismissCacheReminder = document.getElementById('dismissCacheReminder');

    // Function to hide both banners
    const hideAllBanners = () => {
        permissionBanner && (permissionBanner.style.display = 'none');
        cacheReminderBanner && (cacheReminderBanner.style.display = 'none');
    };

    if (permissionBanner && acceptPermissionBanner && declinePermissionBanner && cacheReminderBanner && dismissCacheReminder) {
        const permissionStatus = localStorage.getItem('permissionAccepted');

        if (!permissionStatus) { // No status, show main banner
            permissionBanner.style.display = 'flex';
        } else if (permissionStatus === 'declined') { // Declined previously, show small reminder
            cacheReminderBanner.style.display = 'flex';
        }

        acceptPermissionBanner.addEventListener('click', () => {
            localStorage.setItem('permissionAccepted', 'true');
            hideAllBanners();
        });

        declinePermissionBanner.addEventListener('click', () => {
            localStorage.setItem('permissionAccepted', 'declined');
            hideAllBanners();
            cacheReminderBanner.style.display = 'flex'; // Show small reminder
        });

        dismissCacheReminder.addEventListener('click', () => {
            // User dismissed the reminder, so we won't show it again unless they clear local storage
            localStorage.setItem('cacheReminderDismissed', 'true');
            cacheReminderBanner.style.display = 'none';
        });

        // If reminder was dismissed previously, don't show it again
        if (localStorage.getItem('cacheReminderDismissed') === 'true') {
            cacheReminderBanner.style.display = 'none';
        }
    }

    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('nav-menu');

    // --- Hamburger menu toggle logic ---
    if (hamburger && navMenu) { // Ensure elements exist
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');

            if (navMenu.classList.contains('hidden')) {
                navMenu.classList.remove('hidden');
                navMenu.offsetHeight;
                navMenu.classList.remove('max-h-0');
                navMenu.classList.add('max-h-screen');
            } else {
                navMenu.classList.remove('max-h-screen');
                navMenu.classList.add('max-h-0');

                const transitionEndHandler = () => {
                    navMenu.classList.add('hidden');
                    navMenu.removeEventListener('transitionend', transitionEndHandler);
                };
                navMenu.addEventListener('transitionend', transitionEndHandler);
            }
        });

        navMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                if (window.innerWidth < 768) {
                    hamburger.classList.remove('active');
                    navMenu.classList.remove('max-h-screen');
                    navMenu.classList.add('max-h-0');

                    const transitionEndHandler = () => {
                        navMenu.classList.add('hidden');
                        navMenu.removeEventListener('transitionend', transitionEndHandler);
                    };
                    navMenu.addEventListener('transitionend', transitionEndHandler);
                }
            });
        });
    }

    // Directly set up the consultation button to redirect
    const consultationBtn = document.getElementById("consultationBtn");
    if (consultationBtn) {
        consultationBtn.addEventListener('click', () => {
            window.location.href = '../form.html'; // Redirect to form.html
        });
    }
});

// Example usage: showLoadingPage() before a heavy operation or navigation
const showLoadingPage = () => {
    window.location.href = 'loading.html';
};