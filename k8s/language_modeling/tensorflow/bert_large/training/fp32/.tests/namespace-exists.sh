@test "verify namespace $USER_NAME exists" {
  run verify "there is 1 namespace named '"$USER_NAME"'"
  (( $status == 0 ))
}