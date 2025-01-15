'use client'

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { toast } from "@/components/ui/use-toast"

export default function ResumeSuggestions() {
  const [jobTitle, setJobTitle] = useState('')
  const [resumeText, setResumeText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setSuggestions([])

    try {
      const response = await fetch('https://hypercover.replit.app/api/resume-suggestions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ jobTitle, resumeText }),
      })

      if (response.ok) {
        const data = await response.json()
        setSuggestions(data.suggestions)
      } else {
        throw new Error('Failed to get resume suggestions')
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "There was a problem getting resume suggestions. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section className="py-20 bg-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl md:text-4xl font-monument mb-12 text-center">
          Get <span className="text-[#f15b3b]">AI-Powered</span> Resume Insights
        </h2>
        <div className="grid md:grid-cols-2 gap-12">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <Label htmlFor="jobTitle">Desired Job Title</Label>
              <Input
                id="jobTitle"
                value={jobTitle}
                onChange={(e) => setJobTitle(e.target.value)}
                placeholder="e.g., Software Engineer"
                required
                className="bg-white/10 border-white/20 text-white placeholder-white/50"
              />
            </div>
            <div>
              <Label htmlFor="resumeText">Resume Text</Label>
              <Textarea
                id="resumeText"
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder="Paste your resume text here..."
                required
                className="h-40 bg-white/10 border-white/20 text-white placeholder-white/50"
              />
            </div>
            <Button type="submit" disabled={isLoading} className="w-full bg-[#f15b3b] hover:bg-[#f15b3b]/90">
              {isLoading ? 'Analyzing...' : 'Get Insights'}
            </Button>
          </form>
          <div className="border border-white/10 rounded-lg p-6 backdrop-blur-sm">
            <h3 className="text-xl font-semibold mb-4">AI Insights</h3>
            {suggestions.length > 0 ? (
              <ul className="space-y-2">
                {suggestions.map((suggestion, index) => (
                  <li key={index} className="flex items-start">
                    <span className="text-[#f15b3b] mr-2">•</span>
                    <span>{suggestion}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-400">
                Submit your desired job title and resume to get AI-powered insights on how to improve your resume and match it to job requirements.
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}

