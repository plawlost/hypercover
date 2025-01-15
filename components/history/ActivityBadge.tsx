import { Search, Upload, Briefcase, User, CreditCard, LogIn, LogOut } from 'lucide-react'
import { Badge } from "@/components/ui/badge"

interface ActivityBadgeProps {
  type: string
}

export function ActivityBadge({ type }: ActivityBadgeProps) {
  const badgeConfig = {
    linkedin_search: { icon: Search, label: 'LinkedIn Search', color: 'bg-blue-500' },
    resume_upload: { icon: Upload, label: 'Resume Upload', color: 'bg-green-500' },
    job_suggestion: { icon: Briefcase, label: 'Job Suggestion', color: 'bg-yellow-500' },
    profile_update: { icon: User, label: 'Profile Update', color: 'bg-purple-500' },
    subscription_change: { icon: CreditCard, label: 'Subscription Change', color: 'bg-pink-500' },
    login: { icon: LogIn, label: 'Login', color: 'bg-cyan-500' },
    logout: { icon: LogOut, label: 'Logout', color: 'bg-red-500' },
  }[type] || { icon: Search, label: 'Unknown', color: 'bg-gray-500' }

  const Icon = badgeConfig.icon

  return (
    <Badge variant="secondary" className={`${badgeConfig.color} text-white`}>
      <Icon className="w-4 h-4 mr-1" />
      {badgeConfig.label}
    </Badge>
  )
}

