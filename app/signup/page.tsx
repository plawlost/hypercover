'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { toast } from "@/components/ui/use-toast"

export default function SignUpPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [verificationSent, setVerificationSent] = useState(false)
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      const response = await fetch('https://hypercover.replit.app/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      })

      if (response.ok) {
        setVerificationSent(true)
        toast({
          title: "Success!",
          description: "Your account has been created. Please check your email for verification.",
        })
      } else {
        throw new Error('Failed to register')
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "There was a problem creating your account. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleResendVerification = async () => {
    setIsLoading(true)

    try {
      const response = await fetch('https://hypercover.replit.app/api/auth/resend-confirmation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      })

      if (response.ok) {
        toast({
          title: "Success!",
          description: "A new verification email has been sent. Please check your inbox.",
        })
      } else {
        throw new Error('Failed to resend verification email')
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "There was a problem resending the verification email. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-black">
      <div className="w-full max-w-md space-y-8 rounded-xl bg-white/10 p-8 backdrop-blur-sm">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-white">Create your account</h2>
        </div>
        {!verificationSent ? (
          <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
            <div className="space-y-4 rounded-md shadow-sm">
              <div>
                <Label htmlFor="email-address" className="text-white">Email address</Label>
                <Input
                  id="email-address"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="bg-white/20 text-white placeholder-white/50"
                />
              </div>
              <div>
                <Label htmlFor="password" className="text-white">Password</Label>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="new-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-white/20 text-white placeholder-white/50"
                />
              </div>
            </div>

            <div>
              <Button
                type="submit"
                className="w-full bg-[#f15b3b] hover:bg-[#f15b3b]/90"
                disabled={isLoading}
              >
                {isLoading ? 'Signing up...' : 'Sign up'}
              </Button>
            </div>
          </form>
        ) : (
          <div className="mt-8 space-y-6">
            <p className="text-white text-center">
              A verification email has been sent to {email}. Please check your inbox and click the verification link.
            </p>
            <Button
              onClick={handleResendVerification}
              className="w-full bg-[#f15b3b] hover:bg-[#f15b3b]/90"
              disabled={isLoading}
            >
              {isLoading ? 'Resending...' : 'Resend Verification Email'}
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}

