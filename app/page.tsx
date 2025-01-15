'use client'

import { useState } from 'react'
import { Inter, Space_Grotesk } from 'next/font/google'
import { ArrowRight, Check, Menu, X } from 'lucide-react'
import Link from 'next/link'
import ResumeSuggestions from '@/components/ResumeSuggestions'
import { Button } from "@/components/ui/button"

const inter = Inter({ subsets: ['latin'] })
const spaceGrotesk = Space_Grotesk({ subsets: ['latin'], variable: '--font-space-grotesk' })

export default function HomePage() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  return (
    <div className={`min-h-screen bg-black text-white ${inter.className} ${spaceGrotesk.variable}`}>
      {/* Header */}
      <header className="sticky top-0 z-50 bg-black/80 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <Link href="/" className="flex items-center gap-3">
                <svg
                  viewBox="0 0 317.48 396.67"
                  className="w-8 h-10"
                  aria-hidden="true"
                >
                  <defs>
                    <clipPath id="clippath">
                      <rect className="fill-none" x=".07" width="317" height="396.67"/>
                    </clipPath>
                  </defs>
                  <g clipPath="url(#clippath)">
                    <path
                      className="fill-[#f15b3b]"
                      d="m219.52.36H53.29C23.91.36,0,24.27,0,53.65v289.4c0,29.38,23.91,53.28,53.29,53.28h210.92c29.38,0,53.28-23.9,53.28-53.28V109.85L219.52.36Zm7.77,50.8l44.75,50.02h-44.75v-50.02Zm36.91,317.08H53.29c-13.9,0-25.2-11.3-25.2-25.2V53.65c0-13.9,11.3-25.2,25.2-25.2h145.92v100.81h90.19v213.78c0,13.9-11.31,25.2-25.2,25.2Z"
                    />
                  </g>
                  <g>
                    <path className="fill-[#f15b3b]" d="m58.23,272.59h201.02v28.08H58.23v-28.08Z"/>
                    <path className="fill-[#f15b3b]" d="m58.14,205.15h201.02v28.08H58.14v-28.08Z"/>
                    <path className="fill-[#f15b3b]" d="m141.7,298.35v-86.27h28.08v86.27h-28.08Z"/>
                  </g>
                </svg>
                <span className="font-monument text-sm tracking-wide">HyperCover</span>
              </Link>
            </div>
            <nav className="hidden md:flex space-x-4">
              <Link href="#features" className="text-sm font-medium hover:text-[#f15b3b] transition-colors">Features</Link>
              <Link href="#demo" className="text-sm font-medium hover:text-[#f15b3b] transition-colors">Demo</Link>
              <Link href="#pricing" className="text-sm font-medium hover:text-[#f15b3b] transition-colors">Pricing</Link>
            </nav>
            <div className="hidden md:flex space-x-4">
              <Button asChild variant="ghost" size="sm" className="text-white hover:text-[#f15b3b] transition-colors">
                <Link href="/login">Log In</Link>
              </Button>
              <Button asChild size="sm">
                <Link href="/signup">Sign Up</Link>
              </Button>
            </div>
            <div className="md:hidden">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                aria-label="Toggle menu"
              >
                {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </Button>
            </div>
          </div>
        </div>
        {isMenuOpen && (
          <div className="md:hidden">
            <nav className="px-2 pt-2 pb-3 space-y-1">
              <Link href="#features" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-gray-900 hover:text-[#f15b3b] transition-colors">Features</Link>
              <Link href="#demo" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-gray-900 hover:text-[#f15b3b] transition-colors">Demo</Link>
              <Link href="#pricing" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-gray-900 hover:text-[#f15b3b] transition-colors">Pricing</Link>
              <Link href="/login" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-gray-900 hover:text-[#f15b3b] transition-colors">Log In</Link>
              <Link href="/signup" className="block px-3 py-2 rounded-md text-base font-medium bg-[#f15b3b] text-white hover:bg-[#f15b3b]/90 transition-colors">Sign Up</Link>
            </nav>
          </div>
        )}
      </header>

      <main>
        {/* Hero Section */}
        <section className="py-20 md:py-32">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h1 className="text-4xl md:text-7xl lg:text-8xl font-monument tracking-tight leading-none mb-8">
              <div className="opacity-90">HOURS OF <span className="text-[#f15b3b]">WORK</span>.</div>
              <div className="opacity-90">MOMENTS OF <span className="text-[#f15b3b]">MAGIC</span>.</div>
            </h1>
            <p className="text-xl md:text-2xl mb-12 max-w-3xl">
              HyperCover uses advanced AI to generate personalized cover letters, saving you time and increasing your chances of landing your dream job.
            </p>
            <Button asChild size="lg" className="bg-[#f15b3b] hover:bg-[#f15b3b]/90">
              <Link href="/signup">Start Creating Cover Letters Now</Link>
            </Button>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="py-20 bg-white/5">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl md:text-4xl font-monument mb-12 text-center">
              Features that <span className="text-[#f15b3b]">Revolutionize</span> Your Job Search
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="border border-white/10 rounded-lg p-6 backdrop-blur-sm">
                <h3 className="text-xl font-semibold mb-4">AI-Powered Resume Analysis</h3>
                <p>Our advanced AI analyzes your resume to identify key skills and experiences.</p>
              </div>
              <div className="border border-white/10 rounded-lg p-6 backdrop-blur-sm">
                <h3 className="text-xl font-semibold mb-4">Intelligent Job Matching</h3>
                <p>Get personalized job recommendations based on your unique profile and career goals.</p>
              </div>
              <div className="border border-white/10 rounded-lg p-6 backdrop-blur-sm">
                <h3 className="text-xl font-semibold mb-4">Tailored Cover Letters</h3>
                <p>Generate customized cover letters that highlight your strengths for each job application.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Demo Section */}
        <section id="demo" className="py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl md:text-4xl font-monument mb-12 text-center">
              See HyperCover in <span className="text-[#f15b3b]">Action</span>
            </h2>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div className="space-y-4">
                <div className="border border-[#f15b3b]/20 rounded-lg p-6 bg-black/50">
                  <div className="w-full bg-transparent border-b border-white/10 py-2 px-1 text-white">
                    Upload your CV
                  </div>
                  <div className="mt-6 space-y-4">
                    {['Marketing Manager', 'Software Engineer', 'Data Analyst', 'Sales Representative'].map((role) => (
                      <div key={role} className="flex items-center justify-between">
                        <div className="flex items-center gap-4 text-sm">
                          <div className="w-2 h-2 rounded-full bg-[#f15b3b]" />
                          {role}
                        </div>
                        <button className="text-sm text-[#f15b3b] hover:underline">
                          Generate A Cover Letter
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="space-y-6">
                <h3 className="text-2xl font-semibold">How It Works:</h3>
                <ol className="space-y-4">
                  {[
                    "Upload your CV or resume to HyperCover.",
                    "Select the job roles you're interested in.",
                    "Our AI generates personalized cover letters in seconds.",
                    "Review, edit, and send your cover letters with confidence!"
                  ].map((step, index) => (
                    <li key={index} className="flex items-start">
                      <span className="flex items-center justify-center w-8 h-8 rounded-full bg-[#f15b3b] text-white font-bold mr-4">
                        {index + 1}
                      </span>
                      <p>{step}</p>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          </div>
        </section>

        {/* Resume Suggestions */}
        <ResumeSuggestions />


        {/* Pricing Section */}
        <section id="pricing" className="py-20 bg-white/5">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl md:text-4xl font-monument mb-12 text-center">
              Simple, <span className="text-[#f15b3b]">Affordable</span> Pricing
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  title: "Basic",
                  price: "$9",
                  description: "One-time payment",
                  features: [
                    "Up to 100 cover letters",
                    "Basic AI personalization",
                    "Email support"
                  ]
                },
                {
                  title: "Pro",
                  price: "$19",
                  description: "per month",
                  features: [
                    "Unlimited cover letters",
                    "Advanced AI personalization",
                    "Priority email support",
                    "Resume optimization tips"
                  ],
                  popular: true
                },
                {
                  title: "Lifetime",
                  price: "$99",
                  description: "One-time payment",
                  features: [
                    "Unlimited lifetime usage",
                    "All Pro features",
                    "Future updates included",
                    "Premium support"
                  ]
                }
              ].map((plan, index) => (
                <div key={index} className={`border ${plan.popular ? 'border-[#f15b3b]' : 'border-white/10'} rounded-lg p-8 backdrop-blur-sm relative`}>
                  {plan.popular && (
                    <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-[#f15b3b] text-white px-4 py-1 rounded-full text-sm font-semibold">
                      Most Popular
                    </div>
                  )}
                  <h3 className="text-xl font-semibold mb-4">{plan.title}</h3>
                  <p className="text-3xl font-bold mb-2">{plan.price}</p>
                  <p className="text-sm text-gray-400 mb-6">{plan.description}</p>
                  <ul className="space-y-2 mb-8">
                    {plan.features.map((feature, featureIndex) => (
                      <li key={featureIndex} className="flex items-center">
                        <Check className="w-5 h-5 text-[#f15b3b] mr-2" />
                        {feature}
                      </li>
                    ))}
                  </ul>
                  <Button asChild className={`w-full ${plan.popular ? 'bg-[#f15b3b] hover:bg-[#f15b3b]/90' : ''}`}>
                    <Link href="/signup">Get Started</Link>
                  </Button>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Testimonials */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h2 className="text-3xl md:text-4xl font-monument mb-12 text-center">
              What Our Users <span className="text-[#f15b3b]">Say</span>
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  quote: "HyperCover saved me so much time in my job search. I landed interviews at top companies thanks to the personalized cover letters!",
                  author: "Sarah K., Software Engineer"
                },
                {
                  quote: "As a recent graduate, I was struggling with writing cover letters. HyperCover made the process easy and helped me stand out to employers.",
                  author: "Alex M., Marketing Specialist"
                },
                {
                  quote: "The AI-powered personalization is incredibly accurate. It's like having a professional writer craft each cover letter for you.",
                  author: "David L., Project Manager"
                }
              ].map((testimonial, index) => (
                <div key={index} className="border border-white/10 rounded-lg p-6 backdrop-blur-sm">
                  <p className="mb-4">"{testimonial.quote}"</p>
                  <p className="font-semibold">- {testimonial.author}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA */}
        <section id="get-started" className="py-20 bg-[#f15b3b]">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <h2 className="text-3xl md:text-4xl font-monument mb-8">
              Ready to Transform Your Job Search?
            </h2>
            <p className="text-xl mb-8 max-w-2xl mx-auto">
              Join thousands of job seekers who are landing their dream jobs faster with HyperCover. Start creating impressive, personalized cover letters today!
            </p>
            <Button asChild size="lg" className="bg-white text-[#f15b3b] hover:bg-white/90">
              <Link href="/signup">Sign Up Now</Link>
            </Button>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-black py-12 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              {
                title: "Product",
                links: [
                  { name: "Features", href: "#features" },
                  { name: "Pricing", href: "#pricing" },
                  { name: "FAQ", href: "#" },
                  { name: "API Documentation", href: "#" }
                ]
              },
              {
                title: "Company",
                links: [
                  { name: "About Us", href: "#" },
                  { name: "Careers", href: "#" },
                  { name: "Contact", href: "#" }
                ]
              },
              {
                title: "Resources",
                links: [
                  { name: "Blog", href: "#" },
                  { name: "Guides", href: "#" },
                  { name: "Support", href: "#" }
                ]
              },
              {
                title: "Legal",
                links: [
                  { name: "Privacy Policy", href: "#" },
                  { name: "Terms of Service", href: "#" },
                  { name: "Cookie Policy", href: "#" }
                ]
              }
            ].map((section, index) => (
              <div key={index}>
                <h3 className="text-lg font-semibold mb-4">{section.title}</h3>
                <ul className="space-y-2">
                  {section.links.map((link, linkIndex) => (
                    <li key={linkIndex}>
                      <Link href={link.href} className="hover:text-[#f15b3b] transition-colors">
                        {link.name}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <div className="mt-12 pt-8 border-t border-white/10 text-center">
            <p>&copy; 2023 HyperCover. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}

