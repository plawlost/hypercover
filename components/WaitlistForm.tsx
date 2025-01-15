'use client'

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { toast } from "@/components/ui/use-toast"

export default function WaitlistForm() {
  const [email, setEmail] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      const response = await fetch('https://hypercover.replit.app/api/waitlist', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      })

      if (response.ok) {
        toast({
          title: "Success!",
          description: "You've been added to our waitlist. We'll be in touch soon!",
        })
        setEmail('')
      } else {
        throw new Error('Failed to join waitlist')
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "There was a problem joining the waitlist. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-2 max-w-md mx-auto">
      <Input
        type="email"
        placeholder="Enter your email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
        className="flex-grow bg-white/10 border-white/20 text-white placeholder-white/50"
      />
      <Button type="submit" disabled={isLoading} className="bg-white text-[#f15b3b] hover:bg-white/90">
        {isLoading ? 'Joining...' : 'Join Waitlist'}
      </Button>
    </form>
  )
}

