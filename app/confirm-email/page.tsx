'use client'

import { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { Button } from "@/components/ui/button"
import { toast } from "@/components/ui/use-toast"

export default function ConfirmEmailPage() {
  const [isVerifying, setIsVerifying] = useState(true)
  const [isVerified, setIsVerified] = useState(false)
  const router = useRouter()
  const searchParams = useSearchParams()
  const token = searchParams.get('token')

  useEffect(() => {
    const verifyEmail = async () => {
      if (!token) {
        setIsVerifying(false)
        return
      }

      try {
        const response = await fetch('https://hypercover.replit.app/api/auth/confirm-email', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ token }),
        })

        if (response.ok) {
          setIsVerified(true)
          toast({
            title: "Success!",
            description: "Your email has been verified. You can now log in.",
          })
        } else {
          throw new Error('Failed to verify email')
        }
      } catch (error) {
        toast({
          title: "Error",
          description: "There was a problem verifying your email. The link may have expired.",
          variant: "destructive",
        })
      } finally {
        setIsVerifying(false)
      }
    }

    verifyEmail()
  }, [token])

  return (
    <div className="flex min-h-screen items-center justify-center bg-black">
      <div className="w-full max-w-md space-y-8 rounded-xl bg-white/10 p-8 backdrop-blur-sm text-white text-center">
        {isVerifying ? (
          <p>Verifying your email...</p>
        ) : isVerified ? (
          <>
            <h2 className="text-2xl font-bold mb-4">Email Verified!</h2>
            <p className="mb-6">Your email has been successfully verified. You can now log in to your account.</p>
            <Button
              onClick={() => router.push('/login')}
              className="w-full bg-[#f15b3b] hover:bg-[#f15b3b]/90"
            >
              Go to Login
            </Button>
          </>
        ) : (
          <>
            <h2 className="text-2xl font-bold mb-4">Verification Failed</h2>
            <p className="mb-6">We couldn't verify your email. The verification link may have expired or is invalid.</p>
            <Button
              onClick={() => router.push('/signup')}
              className="w-full bg-[#f15b3b] hover:bg-[#f15b3b]/90"
            >
              Back to Sign Up
            </Button>
          </>
        )}
      </div>
    </div>
  )
}

