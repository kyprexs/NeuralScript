/**
 * NeuralScript Website JavaScript
 * Interactive features and animations
 */

// DOM utilities
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
    initializeComponents();
});

/**
 * Initialize all interactive components
 */
function initializeComponents() {
    initializeNavigation();
    initializeTabs();
    initializeCopyButtons();
    initializeScrollAnimations();
    initializePerformanceCounters();
    initializeSmoothScrolling();
    initializeMobileMenu();
    initializeTypewriter();
    initializeParticleBackground();
}

/**
 * Navigation functionality
 */
function initializeNavigation() {
    const navbar = $('.navbar');
    let lastScrollY = window.scrollY;

    // Update navbar on scroll
    window.addEventListener('scroll', () => {
        const currentScrollY = window.scrollY;
        
        // Hide/show navbar based on scroll direction
        if (currentScrollY > lastScrollY && currentScrollY > 100) {
            navbar.style.transform = 'translateY(-100%)';
        } else {
            navbar.style.transform = 'translateY(0)';
        }
        
        // Add background blur when scrolled
        if (currentScrollY > 10) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
        
        lastScrollY = currentScrollY;
    });

    // Highlight active navigation item
    const navLinks = $$('.nav-links a');
    const sections = $$('section[id]');
    
    window.addEventListener('scroll', () => {
        const scrollPos = window.scrollY + 100;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');
            
            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                navLinks.forEach(link => link.classList.remove('active'));
                const activeLink = $(`.nav-links a[href="#${sectionId}"]`);
                if (activeLink) activeLink.classList.add('active');
            }
        });
    });
}

/**
 * Tab switching functionality
 */
function initializeTabs() {
    const tabButtons = $$('.tab-button');
    const tabPanes = $$('.tab-pane');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.getAttribute('data-tab');
            
            // Remove active class from all buttons and panes
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));
            
            // Add active class to clicked button and corresponding pane
            button.classList.add('active');
            const targetPane = $(`#${targetTab}`);
            if (targetPane) {
                targetPane.classList.add('active');
                
                // Trigger syntax highlighting for the new content
                if (window.Prism) {
                    window.Prism.highlightAllUnder(targetPane);
                }
            }
        });
    });
}

/**
 * Copy to clipboard functionality
 */
function initializeCopyButtons() {
    const copyButtons = $$('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const codeBlock = button.previousElementSibling;
            const text = codeBlock.textContent;
            
            try {
                await navigator.clipboard.writeText(text);
                
                // Visual feedback
                const originalText = button.textContent;
                button.textContent = 'Copied!';
                button.style.background = 'var(--success)';
                button.style.color = 'white';
                
                setTimeout(() => {
                    button.textContent = originalText;
                    button.style.background = '';
                    button.style.color = '';
                }, 2000);
                
            } catch (err) {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            }
        });
    });
}

/**
 * Scroll-triggered animations
 */
function initializeScrollAnimations() {
    const animatedElements = $$('[data-animate]');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                const animation = element.getAttribute('data-animate');
                
                element.classList.add(animation);
                observer.unobserve(element);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    animatedElements.forEach(element => {
        observer.observe(element);
    });

    // Add animation attributes to elements
    const features = $$('.feature-card');
    features.forEach((card, index) => {
        card.setAttribute('data-animate', 'fade-in-up');
        card.style.animationDelay = `${index * 0.1}s`;
    });
    
    const performanceCards = $$('.performance-card');
    performanceCards.forEach((card, index) => {
        card.setAttribute('data-animate', 'fade-in-up');
        card.style.animationDelay = `${index * 0.2}s`;
    });
}

/**
 * Animated performance counters
 */
function initializePerformanceCounters() {
    const counters = $$('.stat-number');
    
    const animateCounter = (element) => {
        const target = element.textContent;
        const numericTarget = parseFloat(target);
        const suffix = target.replace(/[\d.]/g, '');
        
        if (isNaN(numericTarget)) return;
        
        const duration = 2000;
        const steps = 60;
        const increment = numericTarget / steps;
        let current = 0;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= numericTarget) {
                current = numericTarget;
                clearInterval(timer);
            }
            
            element.textContent = current.toFixed(1) + suffix;
        }, duration / steps);
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    });
    
    counters.forEach(counter => observer.observe(counter));
}

/**
 * Smooth scrolling for anchor links
 */
function initializeSmoothScrolling() {
    const links = $$('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            const targetId = link.getAttribute('href').slice(1);
            const targetElement = $(`#${targetId}`);
            
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Mobile menu functionality
 */
function initializeMobileMenu() {
    const mobileMenuButton = $('.mobile-menu');
    const navLinks = $('.nav-links');
    let isOpen = false;
    
    if (!mobileMenuButton) return;
    
    mobileMenuButton.addEventListener('click', () => {
        isOpen = !isOpen;
        
        // Toggle mobile menu styles
        if (isOpen) {
            navLinks.style.display = 'flex';
            navLinks.style.position = 'absolute';
            navLinks.style.top = '100%';
            navLinks.style.left = '0';
            navLinks.style.right = '0';
            navLinks.style.backgroundColor = 'var(--bg-secondary)';
            navLinks.style.flexDirection = 'column';
            navLinks.style.padding = 'var(--space-4)';
            navLinks.style.borderTop = '1px solid var(--border)';
            
            // Animate hamburger menu
            const spans = mobileMenuButton.querySelectorAll('span');
            spans[0].style.transform = 'rotate(45deg) translate(6px, 6px)';
            spans[1].style.opacity = '0';
            spans[2].style.transform = 'rotate(-45deg) translate(6px, -6px)';
        } else {
            navLinks.style.display = 'none';
            
            // Reset hamburger menu
            const spans = mobileMenuButton.querySelectorAll('span');
            spans[0].style.transform = '';
            spans[1].style.opacity = '';
            spans[2].style.transform = '';
        }
    });
    
    // Close mobile menu when clicking outside
    document.addEventListener('click', (e) => {
        if (isOpen && !mobileMenuButton.contains(e.target) && !navLinks.contains(e.target)) {
            mobileMenuButton.click();
        }
    });
    
    // Close mobile menu on window resize
    window.addEventListener('resize', () => {
        if (window.innerWidth >= 768 && isOpen) {
            mobileMenuButton.click();
        }
    });
}

/**
 * Typewriter effect for hero section
 */
function initializeTypewriter() {
    const typewriterElement = $('.gradient-text');
    if (!typewriterElement) return;
    
    const texts = ['Scientific Computing', 'Machine Learning', 'Mathematical Modeling'];
    let textIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    
    function typeWriter() {
        const currentText = texts[textIndex];
        
        if (isDeleting) {
            typewriterElement.textContent = currentText.substring(0, charIndex - 1);
            charIndex--;
        } else {
            typewriterElement.textContent = currentText.substring(0, charIndex + 1);
            charIndex++;
        }
        
        let typeSpeed = isDeleting ? 100 : 150;
        
        if (!isDeleting && charIndex === currentText.length) {
            typeSpeed = 2000; // Pause at end
            isDeleting = true;
        } else if (isDeleting && charIndex === 0) {
            isDeleting = false;
            textIndex = (textIndex + 1) % texts.length;
            typeSpeed = 500; // Pause before starting new text
        }
        
        setTimeout(typeWriter, typeSpeed);
    }
    
    // Start typewriter effect after a delay
    setTimeout(typeWriter, 1000);
}

/**
 * Particle background effect
 */
function initializeParticleBackground() {
    const hero = $('.hero');
    if (!hero) return;
    
    // Create canvas for particles
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.opacity = '0.6';
    canvas.style.zIndex = '0';
    
    hero.appendChild(canvas);
    
    let particles = [];
    
    function resizeCanvas() {
        canvas.width = hero.offsetWidth;
        canvas.height = hero.offsetHeight;
    }
    
    function createParticle() {
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 2 + 1,
            opacity: Math.random() * 0.5 + 0.2
        };
    }
    
    function updateParticles() {
        particles.forEach(particle => {
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Wrap around screen
            if (particle.x < 0) particle.x = canvas.width;
            if (particle.x > canvas.width) particle.x = 0;
            if (particle.y < 0) particle.y = canvas.height;
            if (particle.y > canvas.height) particle.y = 0;
        });
    }
    
    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        particles.forEach(particle => {
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(59, 130, 246, ${particle.opacity})`;
            ctx.fill();
        });
        
        // Draw connections
        particles.forEach((particle, i) => {
            particles.slice(i + 1).forEach(otherParticle => {
                const distance = Math.sqrt(
                    Math.pow(particle.x - otherParticle.x, 2) +
                    Math.pow(particle.y - otherParticle.y, 2)
                );
                
                if (distance < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particle.x, particle.y);
                    ctx.lineTo(otherParticle.x, otherParticle.y);
                    ctx.strokeStyle = `rgba(59, 130, 246, ${0.1 * (1 - distance / 100)})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            });
        });
    }
    
    function animate() {
        updateParticles();
        drawParticles();
        requestAnimationFrame(animate);
    }
    
    // Initialize particles
    resizeCanvas();
    for (let i = 0; i < 50; i++) {
        particles.push(createParticle());
    }
    
    animate();
    
    // Handle resize
    window.addEventListener('resize', resizeCanvas);
}

/**
 * Performance optimization utilities
 */
function throttle(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

/**
 * Keyboard navigation and accessibility
 */
function initializeAccessibility() {
    // Add keyboard navigation for tabs
    const tabButtons = $$('.tab-button');
    tabButtons.forEach((button, index) => {
        button.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                const prevIndex = index === 0 ? tabButtons.length - 1 : index - 1;
                tabButtons[prevIndex].focus();
                tabButtons[prevIndex].click();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                const nextIndex = index === tabButtons.length - 1 ? 0 : index + 1;
                tabButtons[nextIndex].focus();
                tabButtons[nextIndex].click();
            }
        });
    });
    
    // Add skip to content link
    const skipLink = document.createElement('a');
    skipLink.href = '#main-content';
    skipLink.textContent = 'Skip to main content';
    skipLink.className = 'sr-only';
    skipLink.style.position = 'absolute';
    skipLink.style.top = '-40px';
    skipLink.style.left = '6px';
    skipLink.style.zIndex = '1000';
    skipLink.style.background = 'var(--primary)';
    skipLink.style.color = 'white';
    skipLink.style.padding = '8px';
    skipLink.style.textDecoration = 'none';
    skipLink.style.borderRadius = '4px';
    
    skipLink.addEventListener('focus', () => {
        skipLink.style.top = '6px';
    });
    
    skipLink.addEventListener('blur', () => {
        skipLink.style.top = '-40px';
    });
    
    document.body.insertBefore(skipLink, document.body.firstChild);
}

// Initialize accessibility features
document.addEventListener('DOMContentLoaded', initializeAccessibility);

/**
 * Error handling and fallbacks
 */
window.addEventListener('error', (e) => {
    console.warn('JavaScript error caught:', e.error);
    // Graceful degradation - ensure basic functionality still works
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        throttle,
        debounce,
        $,
        $$
    };
}
